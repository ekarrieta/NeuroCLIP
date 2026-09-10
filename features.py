import torch
import numpy as np
import os
from pathlib import Path
from scipy.signal import resample
from torch.nn import functional as F
from scipy.ndimage import gaussian_filter1d
import pickle
import logging
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import warnings
warnings.filterwarnings(
    "ignore",
    message=r"TypedStorage is deprecated.*",
    category=UserWarning,
)

class FeatureExtractor:
    """Feature extraction pipeline for audio and speech processing."""

    def __init__(
        self,
        feature_type: str = "wav",
        model_name: str = "facebook/wav2vec2-large-xlsr-53",
        device: str = 'cpu',
        feature_dim: int = 1024,
        segment_length: float = 3.0,
        cache_dir: str = Path("./cache"),
        ):
        self.device = device
        self.feature_type = feature_type
        self.feature_dim = feature_dim
        self.segment_length = segment_length
        self.cache = Path(cache_dir)
        self.feature_cache = {}

        if feature_type in ["wav2vec2", "flipped_wav2vec2"]:
            self.init_speech_model(model_name)
        elif feature_type in ["word_embeddings", "anchor_word", "sentence_embeddings"]:
            self.init_openai_model(model_name, feature_dim)
        elif feature_type in ["static_embeddings"]:
            self.init_fasttext_model()

    def init_speech_model(self, model_name):
        """Initialize speech models"""
        logging.info(f"Initializing speech model {model_name} on device {self.device}")
        from transformers import Wav2Vec2Model, Wav2Vec2FeatureExtractor
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(model_name)
        self.model_sr = self.feature_extractor.sampling_rate
        self.layers = [14, 15, 16, 17, 18]
        self.model.eval()
        self.model.gradient_checkpointing_enable()  # Enable gradient checkpointing
        self.model.to(self.device)
        self.resampler_cache = {}
        logging.info(f"Loaded speech model: {model_name}")

    def init_openai_model(self, model_name, feature_dim):
        """Initialize the OpenAI embedding client and local embedding caches."""
        from openai import OpenAI

        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError(
                "OPENAI_API_KEY is required for OpenAI embedding features."
            )
        self.openai_client = OpenAI(api_key=api_key)
        self.model_name = model_name
        self.word_cache = {}  # Cache for individual word embeddings
        self.sentence_cache = {}  # Cache for sentence embeddings
        self.feature_dim = feature_dim
        self.init_lm_cache(self.feature_dim)

    def init_fasttext_model(self, model_path=None):
        """Initialize fastText model from a .bin file path."""
        import fasttext
        model_path = model_path or os.getenv("FASTTEXT_MODEL_PATH")
        if not model_path:
            raise RuntimeError(
                "FASTTEXT_MODEL_PATH must point to a fastText .bin file "
                "when feature_type='static_embeddings'."
            )
        self.ft_model = fasttext.load_model(model_path)
        self.ft_dim = self.ft_model.get_dimension()
        logging.info(f"Loaded fastText model from {model_path} (dim={self.ft_dim})")

    def init_lm_cache(self, feature_dim):
        # Init empty embedding for empty strings (1-hot vector)
        self.sentence_cache[""] = torch.cat([torch.tensor([[1.0]]), torch.zeros((feature_dim-1, 1))])
        self.word_cache[""] = torch.cat([torch.tensor([[1.0]]), torch.zeros((feature_dim-1, 1))])
        # Load embedding caches if available
        if self.feature_type in ("word_embeddings", "anchor_word"):
            self.load_word_cache(self.cache / "embeddings" / "word_cache.pkl")
        elif self.feature_type == "sentence_embeddings":
            self.load_sentence_cache(self.cache / "embeddings" / "sentence_cache.pkl")

    def init_feature_cache(self, feature):
        cache_dir = self.cache / "features"
        cache_dir.mkdir(parents=True, exist_ok=True)
        file_path = cache_dir / f"{feature}_{self.segment_length}s_cache.pkl"
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.feature_cache = pickle.load(f)
            logging.info(f"{feature} cache loaded from {file_path} ({len(self.feature_cache)} entries)")
        else:
            self.feature_cache = {}
            logging.info(f"No cache file found at {file_path}. Creating new cache.")

    def flip_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """Flip audio tensor along time dimension."""
        return torch.flip(audio, dims=[-1])

    def extract_features(
        self,
        audio: torch.Tensor,
        sr: torch.Tensor,
        target_len: int,
        words: list = None,
        segment_ids: list = None
    ) -> torch.Tensor:
        """Extract features based on feature_type.

        Args:
            audio: Audio tensor of shape [B, seq] or [seq]
            sr: Sample rate of audio: [B] or float
            target_len: Target length for output features

        Returns:
            Features of shape [B, target_len] or [target_len]
        """

        # Handle both batched and single inputs
        is_batched = audio.dim() == 2
        if not is_batched:
            audio = audio.unsqueeze(0)  # [seq] -> [1, seq]

        if isinstance(sr, torch.Tensor):
            sr = int(sr[0].item())
        else:
            sr = int(sr)

        if self.feature_type == "wav":
            features = self.interpolate_wav(audio, target_len)
        elif self.feature_type == "wav2vec2":
            features = self.extract_speech_features(audio, sr, target_len)
        elif self.feature_type == "flipped_wav2vec2":
            features = self.extract_speech_features(self.flip_audio(audio), sr, target_len)
        elif self.feature_type == "vad":
            features = self.extract_vad(audio, sr, target_len)
        elif self.feature_type == "pitch":
            features = self.extract_pitch(audio, sr, target_len, segment_ids=segment_ids, device=self.device)
        elif self.feature_type == "envelope":
            features = self.extract_envelope(audio, sr, target_len, segment_ids=segment_ids, device=self.device)
        elif self.feature_type == "zcr":
            features = self.extract_zcr(audio, sr, target_len)
        elif self.feature_type == "rms":
            features = self.extract_rms(audio, sr, target_len)
        elif self.feature_type == "noise":
            features = self.return_noise(audio, sr, target_len)
        elif self.feature_type == "mfcc":
            features = self.extract_mfcc(audio, sr, target_len, interpolate=True)
        elif self.feature_type == "mel_spectrogram":
            features = self.extract_mel_spectrogram(audio, sr, target_len, interpolate=True)
        elif self.feature_type == "sentence_embeddings":
            features = self.retrieve_sentence_embeddings(words, target_len)
        elif self.feature_type in ("word_embeddings", "anchor_word"):
            features = self.retrieve_word_embeddings(words, target_len)
        elif self.feature_type == "static_embeddings":
            features = self.extract_static_embeddings(words, target_len)
        else:
            raise ValueError(f"Unsupported feature_type: {self.feature_type}")
        return features

    # ----------Feature extractors-------------

    def interpolate_wav(
        self,
        audio: torch.Tensor,
        target_len: int
        ) -> torch.Tensor:
        """Batch interpolate audio to target_len."""
        # Input: [B, seq_len] -> need [B, 1, seq_len] for interpolate
        # Output: [B, 1, target_len] -> squeeze to [B, target_len]

        assert isinstance(audio, torch.Tensor)

        out = F.interpolate(
            audio.unsqueeze(1),
            size=target_len,
            mode='linear',
            align_corners=False
        ).squeeze(1)
        return out

    def extract_speech_features(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int
    ) -> torch.Tensor:
        """Extract wav2vec2 features from audio"""

        assert isinstance(audio, torch.Tensor)
        import julius

        out_channels = self.model.config.hidden_size
        batch_size = audio.shape[0]

        # Ensure correct sample rate
        model_sr = self.feature_extractor.sampling_rate
        if sr != model_sr:
            if sr not in self.resampler_cache:
                self.resampler_cache[sr] = julius.resample.ResampleFrac(old_sr=int(sr), new_sr=model_sr)
            # Resample each item in batch
            audio = torch.stack([self.resampler_cache[sr](audio[i]) for i in range(batch_size)])

        # Pass the features through the model in minibatches to save memory
        minibatch_size = 32 if batch_size > 32 else batch_size
        all_outs = torch.empty((batch_size, out_channels, target_len), dtype=torch.float32)

        for i in range(0, batch_size, minibatch_size):
            minibatch = audio[i:i+minibatch_size]
            # Feature extractor
            inputs = self.feature_extractor(
                    [minibatch[j].numpy() for j in range(minibatch.shape[0])],
                    return_tensors="pt",
                    sampling_rate=model_sr,
                    do_normalize=True,
                    padding=True
                ).input_values.to(self.device)

            with torch.no_grad():
                outputs = self.model(inputs, output_hidden_states=True)

            out = outputs.hidden_states
            out = torch.stack([out[i] for i in self.layers]).mean(0) # [B, time, features]
            out = out.transpose(1,2)  # [B, features, time]

            out = F.interpolate(
                out,
                size=target_len,
                mode='linear',
                align_corners=False) # [B, features, target_len]

            all_outs[i:i+minibatch_size] = out.cpu()
            # Clear cache
            del inputs, outputs, out
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return all_outs

    def extract_vad(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int
        ) -> torch.Tensor:
        """Extract VAD for batch."""

        assert isinstance(audio, torch.Tensor)

        audio_np = audio.cpu().numpy()
        batch_size = audio_np.shape[0]
        result = np.empty((batch_size, target_len), dtype=np.float32)

        for i in range(batch_size):
            result[i] = self.simple_vad(
                audio_np[i], sr, target_len)
        return torch.from_numpy(result)

    def simple_vad(
        self,
        audio: np.ndarray,
        sr: int,
        target_len: int,
        frame_ms: int = 30
        ) -> np.ndarray:
        """Simple energy-based VAD with adaptive threshold."""

        frame_size = int(sr * frame_ms / 1000)
        n_frames = len(audio) // frame_size

        energies = np.array([
            np.sum(audio[i*frame_size:(i+1)*frame_size]**2) / frame_size
            for i in range(n_frames)
        ])
        threshold = max(energies.mean() * 0.1, 1e-10)

        vad_signal = []
        for i in range(n_frames):
            vad_signal.extend([1.0 if energies[i] > threshold else 0.0] * frame_size)

        # Pad remaining
        remaining = len(audio) - len(vad_signal)
        if remaining > 0:
            vad_signal.extend([0.0] * remaining)

        vad_signal = np.array(vad_signal[:len(audio)])
        # Smooth before resampling
        vad_signal = gaussian_filter1d(vad_signal, sigma=sr*0.008)  # 8ms smoothing
        return resample(vad_signal, target_len)

    def extract_zcr(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int,
        frame_ms: int = 25,
        hop_ms: int = 8
        ) -> torch.Tensor:
        """Extract Zero-Crossing Rate for batch"""

        assert isinstance(audio, torch.Tensor)
        import librosa

        audio = audio.cpu().numpy()
        batch_size = audio.shape[0]
        result = np.empty((batch_size, target_len), dtype=np.float32)

        # Calculate frame and hop lengths based on sample rate
        frame_length = int(sr * frame_ms / 1000)
        hop_length = int(sr * hop_ms / 1000)

        for i in range(batch_size):
            # Compute ZCR using librosa
            zcr = librosa.feature.zero_crossing_rate(
                audio[i],
                frame_length=frame_length,
                hop_length=hop_length)[0]

            # Smooth the ZCR signal before resampling
            zcr = gaussian_filter1d(zcr, sigma=1.0)
            # Resample to target length
            result[i] = resample(zcr, target_len)

        return torch.from_numpy(result)

    def extract_rms(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int,
        frame_ms: int = 32,
        hop_ms: int = 8
        ) -> torch.Tensor:
        """Extract Short-Term RMS energy"""

        assert isinstance(audio, torch.Tensor)
        import librosa

        audio = audio.cpu().numpy()
        batch_size = audio.shape[0]
        result = np.empty((batch_size, target_len), dtype=np.float32)

        # Calculate frame and hop lengths based on sample rate
        frame_length = int(sr * frame_ms / 1000)
        hop_length = int(sr * hop_ms / 1000)

        for i in range(batch_size):
            # Compute RMS energy using librosa
            rms = librosa.feature.rms(
                y=audio[i],
                frame_length=frame_length,
                hop_length=hop_length
            )[0]

            # Optional: smooth the RMS signal before resampling
            rms = gaussian_filter1d(rms, sigma=1.0)

            # Resample to target length
            result[i] = resample(rms, target_len)

        return torch.from_numpy(result)

    def extract_mfcc(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int,
        n_mfcc: int = 13,
        n_fft: int = 512,
        hop_length: int = 160,
        norm: bool = False,
        interpolate: bool = False,
    ) -> torch.Tensor:
        """Extract MFCCs for batch.

        Args:
            audio: [B, T] tensor
            sr: sample rate
            target_len: number of output time steps
            n_mfcc: number of MFCC coefficients (default 13)
            n_fft: FFT window size
            hop_length: hop length in samples

        Returns:
            Tensor of shape [B, n_mfcc, target_len]
        """
        assert isinstance(audio, torch.Tensor)
        import librosa

        audio_np = audio.cpu().numpy()
        batch_size = audio_np.shape[0]
        result = np.empty((batch_size, n_mfcc, target_len), dtype=np.float32)

        for i in range(batch_size):
            mfccs = librosa.feature.mfcc(
                y=audio_np[i],
                sr=sr,
                n_mfcc=n_mfcc,
                n_fft=n_fft,
                hop_length=hop_length,
            )  # [n_mfcc, frames]

            # Remove first coefficient (overall energy) to focus on spectral shape
            # mfccs = mfccs[1:]  # [n_mfcc-1, target_len]

            if norm:
                mfccs = (mfccs - mfccs.mean(axis=1, keepdims=True)) / (mfccs.std(axis=1, keepdims=True) + 1e-8)

            # Resample each coefficient to target_len
            mfccs_rs = np.empty((n_mfcc, target_len), dtype=np.float32)
            for c in range(n_mfcc):
                mfccs_rs[c] = resample(mfccs[c], target_len)
            result[i] = mfccs_rs

        # Interpolate the coefficient axis from n_mfcc -> self.feature_dim
        if interpolate:
            result = torch.from_numpy(result)          # [B, n_mfcc, target_len]
            result = result.permute(0, 2, 1)           # [B, target_len, n_mfcc]
            result = F.interpolate(result, size=self.feature_dim, mode='linear', align_corners=False)
            result = result.permute(0, 2, 1)           # [B, feature_dim, target_len]
            return result

        return torch.from_numpy(result)

    def extract_mel_spectrogram(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int,
        n_mels: int = 120,
        n_fft: int = 512,
        hop_length: int = 128,
        mel_sr: int = 16000,
        eps: float = 1e-5,
        interpolate: bool = False,
    ) -> torch.Tensor:
        """Extract paper-style log-compressed mel spectrograms for batch.

        Returns:
            Tensor of shape [B, n_mels, target_len]
        """
        assert isinstance(audio, torch.Tensor)
        import librosa

        audio_np = audio.cpu().float().numpy()
        if sr != mel_sr:
            audio_np = np.stack([
                librosa.resample(y, orig_sr=sr, target_sr=mel_sr)
                for y in audio_np
            ])

        audio_stft = torch.from_numpy(audio_np)
        window = torch.hann_window(n_fft, dtype=audio_stft.dtype)
        stft = torch.stft(
            audio_stft,
            n_fft=n_fft,
            hop_length=hop_length,
            win_length=n_fft,
            window=window,
            center=True,
            normalized=True,
            return_complex=True,
        )
        power = stft.abs().pow(2).numpy()  # [B, freqs, frames]
        mel_basis = librosa.filters.mel(
            sr=mel_sr,
            n_fft=n_fft,
            n_mels=n_mels,
        ).astype(np.float32)
        mel = np.einsum("mf,bft->bmt", mel_basis, power)
        mel = np.log(eps + mel)

        batch_size = mel.shape[0]
        result = np.empty((batch_size, n_mels, target_len), dtype=np.float32)

        for i in range(batch_size):
            mel_rs = np.empty((n_mels, target_len), dtype=np.float32)
            for c in range(n_mels):
                mel_rs[c] = resample(mel[i, c], target_len)
            result[i] = mel_rs

        if interpolate:
            result = torch.from_numpy(result)          # [B, n_mels, target_len]
            result = result.permute(0, 2, 1)           # [B, target_len, n_mels]
            result = F.interpolate(result, size=self.feature_dim, mode="linear", align_corners=False)
            result = result.permute(0, 2, 1)           # [B, feature_dim, target_len]
            return result

        return torch.from_numpy(result)

    def return_noise(
        self,
        audio: torch.Tensor,
        sr: int,
        target_len: int,
        mean: float = 0.0,
        std: float = 1.0
        ) -> torch.Tensor:
        """Generate Gaussian noise"""

        assert isinstance(audio, torch.Tensor)

        batch_size = audio.shape[0]
        # Generate Gaussian noise
        noise = torch.randn(batch_size, target_len) * std + mean
        return noise

    def retrieve_sentence_embeddings(
        self,
        sentences: list,
        target_len: int,
        norm: bool = False
        ) -> torch.Tensor:
        """Create sentence embeddings using OpenAI API with caching."""
        # Check if sentences are cached
        new_sentences = list(dict.fromkeys(s for s in sentences if s not in self.sentence_cache))

        # Fetch embeddings for uncached sentences from OpenAI API in a single batch
        if new_sentences:
            response = self.openai_client.embeddings.create(model=self.model_name,
                                               input=new_sentences,
                                               dimensions=self.feature_dim,
                                               encoding_format="float")
            new_embeddings = [torch.tensor(data.embedding).view(self.feature_dim, 1) for data in response.data]  # Ensure [1024, 1]

            # Cache the embeddings
            for sentence, embedding in zip(new_sentences, new_embeddings):
                self.sentence_cache[sentence] = embedding

        # Retrieve embeddings, and expand to match target_len
        batch_embeddings = []
        for sentence in sentences:
            emb = self.sentence_cache[sentence]
            if norm:
                # Standardize embedding (mean=0, std=1)
                emb = (emb - emb.mean()) / (emb.std() + 1e-8)
            sentence_embedding = emb.expand(-1, target_len)
            batch_embeddings.append(sentence_embedding)

        return torch.stack(batch_embeddings)

    def retrieve_word_embeddings(
        self,
        sentences: list,
        target_len: int
    ) -> torch.Tensor:
        """Create word-level embeddings and concatenate with character-proportional repetition"""

        # Collect all unique words across the batch
        all_words = set()
        for sentence in sentences:
            if sentence:  # Handle empty strings
                all_words.update(sentence.split())

        # Get uncached words
        new_words = sorted(w for w in all_words if w not in self.word_cache)

        # Fetch embeddings for new words using OpenAI API
        if new_words:
            response = self.openai_client.embeddings.create(model=self.model_name,
                                                input=new_words,
                                                dimensions=self.feature_dim,
                                                encoding_format="float")

            # Cache the new embeddings
            for word, data in zip(new_words, response.data):
                embedding = torch.tensor(data.embedding).view(self.feature_dim, 1)
                self.word_cache[word] = embedding

        # Build batch embeddings
        batch_embeddings = []
        for sentence in sentences:
            if not sentence or sentence.strip() == "":
                # Empty sentence - use cached empty embedding
                empty_emb = self.word_cache[""].expand(-1, target_len)
                batch_embeddings.append(empty_emb)
                continue
            words = sentence.split()
            # Calculate total character count (excluding spaces)
            total_chars = sum(len(word) for word in words)
            # Build word embeddings proportional to character length
            word_embeddings = []
            for word in words:
                # Get cached word embedding [feature_dim, 1]
                word_emb = self.word_cache[word]
                # Calculate how many timesteps this word should occupy
                char_proportion = len(word) / total_chars
                repeat_times = int(np.ceil(char_proportion * target_len))
                # Repeat the embedding [feature_dim, repeat_times]
                repeated_emb = word_emb.expand(-1, repeat_times)
                word_embeddings.append(repeated_emb)
            # Concatenate all word embeddings [feature_dim, total_time]
            sentence_embedding = torch.cat(word_embeddings, dim=-1)
            # Resample to exact target_len
            if sentence_embedding.shape[1] != target_len:
                # Add batch dimension for interpolate [1, feature_dim, time]
                sentence_embedding = sentence_embedding.unsqueeze(0)
                sentence_embedding = F.interpolate(sentence_embedding,
                                                    size=target_len,
                                                    mode='linear',
                                                    align_corners=False)
                # Remove batch dimension [feature_dim, target_len]
                sentence_embedding = sentence_embedding.squeeze(0)
            batch_embeddings.append(sentence_embedding)
        # Stack batch [B, feature_dim, target_len]
        return torch.stack(batch_embeddings)

    def extract_static_embeddings(
        self,
        sentences: list,
        target_len: int
    ) -> torch.Tensor:
        """Sentence embeddings by averaging fastText word vectors.

        Returns: [B, feature_dim, target_len]
        """
        batch_embeddings = []
        for sentence in sentences:
            words = sentence.split() if sentence and sentence.strip() else []
            if words:
                vecs = np.stack([self.ft_model.get_word_vector(w) for w in words])  # [n_words, ft_dim]
                emb = vecs.mean(axis=0)  # [ft_dim]
            else:
                emb = np.zeros(self.ft_dim, dtype=np.float32)

            emb = torch.from_numpy(emb).float()       # [ft_dim]
            emb = emb.view(1, 1, self.ft_dim)         # [1, 1, ft_dim]
            # Project ft_dim -> feature_dim
            emb = F.interpolate(emb, size=self.feature_dim, mode='linear', align_corners=False)  # [1, 1, feature_dim]
            emb = emb.squeeze(0).transpose(0, 1)      # [feature_dim, 1]
            emb = emb.expand(-1, target_len)           # [feature_dim, target_len]
            batch_embeddings.append(emb)

        return torch.stack(batch_embeddings)           # [B, feature_dim, target_len]

    def load_sentence_cache(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.sentence_cache = pickle.load(f)
            logging.info(f"Sentence cache loaded from {file_path} ({len(self.sentence_cache)} entries)")
        else:
            logging.info(f"No cache file found at {file_path}. Creating new cache.")

    def save_sentence_cache(self, file_path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.sentence_cache, f)
        logging.info(f"Sentence cache saved to {file_path} ({len(self.sentence_cache)} entries)")

    def load_word_cache(self, file_path):
        if os.path.exists(file_path):
            with open(file_path, 'rb') as f:
                self.word_cache = pickle.load(f)
            logging.info(f"Word cache loaded from {file_path} ({len(self.word_cache)} entries)")
        else:
            logging.info(f"No cache file found at {file_path}. Creating new cache.")

    def save_word_cache(self, file_path):
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with open(file_path, 'wb') as f:
            pickle.dump(self.word_cache, f)
        logging.info(f"Word cache saved to {file_path} ({len(self.word_cache)} entries)")



    @torch.inference_mode()
    def extract_envelope(self,
                              audio: torch.Tensor,
                              sr: int,
                              target_len: int,
                              segment_ids: list = None,
                              device = 'cpu'
                              ) -> torch.Tensor:
        """
        Batched amplitude envelope using Hilbert in PyTorch.
        audio: [B, T] on CPU or CUDA
        returns: [B, target_len] on same device
        """
        B, T = audio.shape
        dtype = audio.dtype

        # Ensure audio is on the target device once up-front
        if audio.device != device:
            audio = audio.to(device, dtype=dtype, non_blocking=True)

        # Initialize result tensor
        result = torch.empty((B, target_len), dtype=dtype, device=device)

        # Track which samples need computation
        compute_mask = torch.ones(B, dtype=torch.bool, device=device)
        # Check cache if segment_ids provided
        if segment_ids is not None and self.feature_cache is not None:
            for i, seg_id in enumerate(segment_ids):
                if seg_id in self.feature_cache:
                    result[i] = self.feature_cache[seg_id].to(device, non_blocking=True)
                    compute_mask[i] = False

        # Compute only uncached samples
        if compute_mask.any():
            audio_compute = audio[compute_mask].contiguous()

            spectrum = torch.fft.fft(audio_compute, n=T, dim=-1)
            h = torch.zeros(T, device=device, dtype=dtype)
            h[0] = 1
            if T % 2 == 0:
                h[T // 2] = 1
                h[1 : T // 2] = 2
            else:
                h[1 : (T + 1) // 2] = 2
            analytic = torch.fft.ifft(spectrum * h, n=T, dim=-1)
            env = analytic.abs()

            # smoothing
            # kernel_size ~ sr*0.008 (8 ms) clipped to odd small window
            k = max(3, int(sr * 0.008) | 1)
            pad = k // 2
            if k > 3:
                t = torch.arange(k, device=device, dtype=dtype) - (k - 1) / 2
                g = torch.exp(-0.5 * (t / (0.3 * k)) ** 2)
                g = (g / g.sum()).view(1, 1, -1)
                env = F.conv1d(env.unsqueeze(1), g, padding=pad).squeeze(1)

            # Resample to target_len via linear interpolate
            env_rs = F.interpolate(env.unsqueeze(1), size=target_len, mode='linear', align_corners=False).squeeze(1)

            # Store computed results and cache them
            compute_idx = 0
            for i in range(B):
                if compute_mask[i]:
                    result[i] = env_rs[compute_idx]
                    # Cache if segment_id provided
                    if segment_ids is not None:
                        self.feature_cache[segment_ids[i]] = env_rs[compute_idx].detach().cpu()
                    compute_idx += 1
        return result

    @torch.inference_mode()
    def extract_pitch(
        self,
        audio: torch.Tensor,      # [B, T] float32 in [-1,1], CPU or CUDA
        sr: int,
        target_len: int,
        segment_ids: list = None,
        model: str = "full", # "tiny" is faster, "full" is more accurate
        hop_length: int = 120,    # 7.5 ms at 16 kHz; we'll resample as needed
        fmin_hz: float = 50.0,
        fmax_hz: float = 550.0,
        silence_db: float = -60.0,
        at_threshold: float = 0.25,
        device = 'cuda'
    ) -> torch.Tensor:
        """
        High-quality, batched F0 using torchcrepe. Returns [B, target_len] on input device.
        """
        import torchcrepe

        dtype = torch.float32
        B, _ = audio.shape

        # Ensure audio is on the target device once up-front
        if audio.device != device:
            audio = audio.to(device, dtype=dtype, non_blocking=True)

        # Initialize result tensor
        result = torch.empty((B, target_len), dtype=dtype, device=device)

        # Track which samples need computation
        compute_mask = torch.ones(B, dtype=torch.bool, device=device)

        # Check cache if segment_ids provided
        if segment_ids is not None and self.feature_cache is not None:
            for i, seg_id in enumerate(segment_ids):
                if seg_id in self.feature_cache:
                    result[i] = self.feature_cache[seg_id].to(device, non_blocking=True)
                    compute_mask[i] = False

        # Compute only uncached samples
        if compute_mask.any():
            audio_compute = audio[compute_mask].contiguous()
            # Process torchcrepe inference in minibatches to control memory use
            minibatch_size = 32
            f0_chunks = []

            for start in range(0, audio_compute.shape[0], minibatch_size):
                end = min(start + minibatch_size, audio_compute.shape[0])
                audio_chunk = audio_compute[start:end]

                # Predict F0 in Hz; returns [B_chunk, frames], periodicity
                f0_chunk, pd_chunk = torchcrepe.predict(
                    audio_chunk,
                    sr,
                    hop_length,
                    fmin=fmin_hz,
                    fmax=fmax_hz,
                    model=model,
                    device=device,
                    return_periodicity=True,
                )

                # Silence threshold. Applied per-sample (expects batch=1)
                silencer = torchcrepe.threshold.Silence(silence_db)
                pd_silenced = []
                for b in range(pd_chunk.shape[0]):
                    pd_b = silencer(
                        pd_chunk[b : b + 1],      # shape [1, frames]
                        audio_chunk[b : b + 1],   # shape [1, T]
                        sr,
                        hop_length,
                    )
                    pd_silenced.append(pd_b)
                pd_chunk = torch.cat(pd_silenced, dim=0)  # back to [B, frames]

                f0_chunk = torchcrepe.filter.median(f0_chunk, 3)
                f0_chunk = torchcrepe.threshold.At(at_threshold)(f0_chunk, pd_chunk)

                # Replace NaNs/Infs with 0 Hz
                f0_chunk = torch.nan_to_num(f0_chunk, nan=0.0, posinf=0.0, neginf=0.0)
                # Resample frames → target_len (linear, differentiable)
                f0_chunk = F.interpolate(
                    f0_chunk.unsqueeze(1),
                    size=target_len,
                    mode="linear",
                    align_corners=False,
                ).squeeze(1)

                f0_chunks.append(f0_chunk)

            f0 = torch.cat(f0_chunks, dim=0)

            # Store computed results and cache them
            compute_idx = 0
            for i in range(B):
                if compute_mask[i]:
                    result[i] = f0[compute_idx]
                    # Cache if segment_id provided
                    if segment_ids is not None:
                        self.feature_cache[segment_ids[i]] = f0[compute_idx].detach().cpu()
                    compute_idx += 1
        return result.to(torch.float32)
