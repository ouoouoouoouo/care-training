"""Drop-in replacement for CARE/pretraining/config.py.

All paths point to /home/ouo/care_training/ — adjust if your layout differs.

Apply via:
    cp care-training/patches/care_config.py /home/ouo/care_training/CARE/pretraining/config.py
(or run patches/apply_care_patches.sh)
"""

# ---- Audio source ----------------------------------------------------------
podcast_audio_folder            = "/home/ouo/dataset/MSP_Podcast/Audios"

# ---- Pre-computed features (Phases 2-4) ------------------------------------
# Acoustic supervision target: we use PASE+ (paper) not OpenSMILE (release code).
# Same path is referenced under both names because dataset_pase.py only reads
# `podcast_pase_feats` in our usage.
podcast_pase_feats              = "/home/ouo/care_training/data/pase_features"
podcast_opensmile_feats         = "/home/ouo/care_training/data/pase_features"  # alias

# Semantic supervision target: RoBERTa-base mean-pool of Whisper transcripts.
podcast_roberta_feats_whisper   = "/home/ouo/care_training/data/roberta_features"
podcast_roberta_feats_whisper_sup = "/home/ouo/care_training/data/roberta_features"   # alias (supervised path; unused unless --supervised)

# RoBERTa logits — we don't generate these; patched dataset loads zeros on missing.
podcast_roberta_logits          = "/home/ouo/care_training/data/roberta_logits_PLACEHOLDER"

# ---- Transcripts JSON (Phase 5) --------------------------------------------
podcast_transcripts             = "/home/ouo/care_training/data/whisper_transcripts.json"

# ---- WavLM tokens (Phase 5 placeholder; patched dataset skips length filter)
podcast_wavlm_tokens            = "/home/ouo/care_training/data/wavlm_tokens.txt"
podcast_wavlm_tokens_6          = "/home/ouo/care_training/data/wavlm_tokens.txt"   # same placeholder

# ---- WavLM continuous features (Phase 3 — optional, not required) ----------
podcast_wavlm_feats             = "/home/ouo/care_training/data/wavlm_features"
podcast_wavlm_feats_6           = "/home/ouo/care_training/data/wavlm_features"

# ---- Train/val splits (Phase 5) --------------------------------------------
train_files                     = "/home/ouo/care_training/data/trainlist.pkl"
valid_files                     = "/home/ouo/care_training/data/vallist.pkl"

# ---- Labels CSVs / JSON ----------------------------------------------------
podcast_labels                  = "/home/ouo/dataset/MSP_Podcast/Labels/labels_consensus.csv"
# Pseudo-labels JSON: {filename.wav: int_label_0/1/2}
# Generate via care-training/scripts/prepare_care_text_labels.py
podcast_text_labels             = "/home/ouo/care_training/data/text_labels.json"

# ---- Unused-by-our-flow paths (kept so any stray import still resolves) ----
podcast_roberta_feats           = "/home/ouo/care_training/data/roberta_features"
podcast_roberta_feats_large     = "/home/ouo/care_training/data/roberta_features"
podcast_roberta_feats_paraphrasings = "/home/ouo/care_training/data/roberta_features"
podcast_roberta_feats_all       = "/home/ouo/care_training/data/roberta_features"
podcast_roberta_feats_supervised = "/home/ouo/care_training/data/roberta_features"

energy_folder                   = "/home/ouo/care_training/data/energy_PLACEHOLDER"
pitch_folder                    = "/home/ouo/care_training/data/pitch_PLACEHOLDER"
quantized_energy_folder         = "/home/ouo/care_training/data/quantized_energy_PLACEHOLDER"
quantized_pitch_folder          = "/home/ouo/care_training/data/quantized_pitch_PLACEHOLDER"

second_stage_semantic           = "/home/ouo/care_training/data/second_stage_semantic"
second_stage_nonsemantic        = "/home/ouo/care_training/data/second_stage_nonsemantic"
