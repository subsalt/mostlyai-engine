# Copyright 2025 MOSTLY AI
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
MostlyAI Engine - Optimized for distributed cluster environments.

This fork provides an in-memory API for training and generating synthetic data
without workspace file I/O.

Training:
    >>> from mostlyai.engine import train_flat, train_sequential
    >>> artifact = train_flat(df, encoding_types, max_epochs=100)
    >>> artifact = train_sequential(events_df, encoding_types, ctx_data=users_df, ...)

Generation:
    >>> from mostlyai.engine import generate_flat, generate_sequential
    >>> synthetic_df = generate_flat(artifact, sample_size=1000)
    >>> synthetic_df = generate_sequential(artifact, sample_size=100)

Serialization:
    >>> artifact_bytes = artifact.to_bytes()  # Save to DB/object store
    >>> artifact = ModelArtifact.from_bytes(artifact_bytes)  # Load back
"""
import warnings

from mostlyai.engine._artifact import ModelArtifact, minimize_stats
from mostlyai.engine._logging import init_logging
from mostlyai.engine._stats import compute_stats
from mostlyai.engine._tabular.training import EpochInfo, OnEpochCallback
from mostlyai.engine._train import train_flat, train_sequential
from mostlyai.engine.domain import ModelEncodingType
from mostlyai.engine.generation import generate_flat, generate_sequential
from mostlyai.engine.random_state import set_random_state

__all__ = [
    # Core API
    "train_flat",
    "train_sequential",
    "generate_flat",
    "generate_sequential",
    # Artifact
    "ModelArtifact",
    "minimize_stats",
    "compute_stats",
    # Types
    "ModelEncodingType",
    "EpochInfo",
    "OnEpochCallback",
    # Utilities
    "init_logging",
    "set_random_state",
]
__version__ = "2.3.3"

# suppress specific warning related to os.fork() in multi-threaded processes
warnings.filterwarnings("ignore", category=DeprecationWarning, message=".*multi-threaded.*fork.*")
