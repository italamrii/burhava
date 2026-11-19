#!/usr/bin/env bash
# Download Stage C datasets into data/raw/ (git-ignored) and verify checksums.
#   ProofWriter V2020.12.3 (Tafjord, Dalvi & Clark 2021), AI2. The archive's README
#     states no licence; check AI2's dataset page before redistributing derived data.
#   PrOntoQA-OOD generated data (Saparov et al. 2023), Apache-2.0 (LICENSE saved alongside).
set -euo pipefail
mkdir -p data/raw/proofwriter data/raw/prontoqa
curl -sS -o data/raw/proofwriter/proofwriter.zip \
  https://aristo-data-public.s3.amazonaws.com/proofwriter/proofwriter-dataset-V2020.12.3.zip
curl -sS -o data/raw/prontoqa/generated_ood_data.zip \
  https://raw.githubusercontent.com/asaparov/prontoqa/main/generated_ood_data.zip
curl -sS -o data/raw/prontoqa/LICENSE https://raw.githubusercontent.com/asaparov/prontoqa/main/LICENSE
sha256sum -c <<'SUMS'
bbc5694901e8306d0bd659aa1ad53ccfd02c201864f4b320ffa3777827d1fc26  data/raw/proofwriter/proofwriter.zip
0becba04e1e1eb80593709c6a3db2a46badd5587a29587ffaeb68ed5068c9de9  data/raw/prontoqa/generated_ood_data.zip
SUMS
mkdir -p data/raw/proofwriter/x data/raw/prontoqa/x
unzip -o -q data/raw/proofwriter/proofwriter.zip \
  'proofwriter-dataset-V2020.12.3/CWA/depth-5/*' 'proofwriter-dataset-V2020.12.3/README.md' \
  -d data/raw/proofwriter/x
unzip -o -q data/raw/prontoqa/generated_ood_data.zip '*ProofsOnly*' -d data/raw/prontoqa/x
echo "data ready"
