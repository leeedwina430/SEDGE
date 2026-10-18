# Third-party notices

## AnyMolGenCritic

The source files under `src/sedge_molecule/_vendor/anymolgencritic/src/` come from [SamsungSAILMontreal/AnyMolGenCritic](https://github.com/SamsungSAILMontreal/AnyMolGenCritic).

Copyright (c) 2024 Samsung Electronics Co., Ltd. All Rights Reserved.

These files are distributed under the MIT License. The complete, unchanged copyright notice, permission grant and warranty disclaimer are retained in [the vendored LICENSE](src/sedge_molecule/_vendor/anymolgencritic/LICENSE).

The upstream source retains its own attribution comments: `model/transformers_sota.py` credits nanoGPT, Hugging Face Transformers rotary embedding code, and MosaicML MPT RMSNorm; `data/target_data.py` credits keras-molecules and Stanford pretrain-gnns as inspirations. The attribution comments retain the repository and source filenames. This package does not relicense the vendored code.

The import bridge in `src/sedge_molecule/upstream.py` is a local packaging adaptation. It resolves the vendor directory relative to the installed package without requiring a nested Git checkout. It does not modify the vendored architecture, token definitions, graph grammar or loss helper.
