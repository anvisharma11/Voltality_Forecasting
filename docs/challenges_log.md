# Project Challenges and Resolutions Log

This log records real technical challenges encountered during setup and model training, their causes, and resolutions.

## Challenge 1: TensorFlow Installation Failure on Windows with Python 3.12
- **What happened:** Attempting to install TensorFlow on Windows under Python 3.12 failed due to an interrupted connection and hash verification error on the 350 MB wheel from PyPI.
- **Why:** Large binary wheels over unstable network connections are prone to partial download hash mismatches on Windows with Python 3.12.
- **How it was fixed:** In accordance with `IMPLEMENTATION_PLAN.md` Rule 8 and Section T0 fallback rules, switched the deep learning framework to PyTorch CPU (`torch==2.14.1+cpu`), which installed cleanly and provides identical LSTM model architecture capabilities (`LSTM(64) -> Dropout(0.2) -> Linear(1)`).

## Challenge 2: SciPy 1.18.1 DLL Blocked by Windows 11 Smart App Control
- **What happened:** SciPy 1.18.1 failed to load with `ImportError: DLL load failed while importing _linalg_pythran: An Application Control policy has blocked this file.`
- **Why:** Windows 11 Smart App Control blocks newly released or unsigned compiled extensions (`.pyd`) in user directories that lack established cloud reputation in Microsoft's Intelligent Security Graph.
- **How it was fixed:** Unblocked downloaded files via PowerShell `Unblock-File` and installed the mature, widely adopted `scipy==1.13.1` (with `numpy==2.2.6`), which is fully recognized by Smart App Control and imports cleanly.
