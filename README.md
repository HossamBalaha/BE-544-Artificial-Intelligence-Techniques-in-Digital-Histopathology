# BE 544 Artificial Intelligence (AI) Techniques in Digital Histopathology (Spring 2026) - Updated

Welcome to the BE 544: Artificial Intelligence (AI) Techniques in Digital Histopathology course.

This course offers both theoretical and practical knowledge about computer vision and AI techniques essential for
processing and analyzing microscopic images, contributing to the shift towards digital pathology. This transition will
allow AI models to assist pathologists and healthcare professionals in managing and diagnosing various diseases.

We will explore how artificial intelligence is revolutionizing bioengineering, particularly in analyzing and
interpreting digital pathology images. Join us as we uncover the latest advancements and methodologies in this exciting
intersection of technology and healthcare. Whether you are a student, researcher, or simply curious about the future of
healthcare technology, this playlist offers valuable insights into the innovative applications of AI in digital
histopathology.

> If you encountered any issues or errors in the code or lectures, please feel free to let me know. I will be more than
> happy to fix them and update the repository accordingly. Your feedback is highly appreciated and will help me improve
> the quality of the content provided in this series.

## Full Playlist and Videos

This series is your gateway to the fascinating world of applying AI techniques to digital histopathology.

**Recent Playlist**:

Link: https://www.youtube.com/playlist?list=PLVrN2LRb7eT3_la39bWC0EP-IW5jNjQ-w

**Earlier Playlists**:

> Playlist from Spring 2025 (AI-Generated Podcasts):
> https://www.youtube.com/playlist?list=PLVrN2LRb7eT0VBZqrtSAJQd2mqVtIDJKx

> Playlist from Summer 2024 (Recorded): https://www.youtube.com/playlist?list=PLVrN2LRb7eT2KV3YMdXeF2B9dgaN4QF4g

## Programming Language and Libraries

This project is written in Python. All Python package dependencies required by the lectures and examples are listed in
the `requirements.txt` file at the repository root.

You can install the dependencies directly with pip (system / virtualenv / activated conda env):

```cmd
pip install -r requirements.txt
```

Recommended Python version: Python `3.10` (the materials were developed and tested with Python `3.10.x`, e.g. `3.10.18`)
on a Windows machine. The code will often work with other Python `3.10.x` builds, but behavior on other Python
major/minor versions or on other operating systems has not been exhaustively tested.

## Anaconda Environment Setup (Optional But Recommended)

A helper batch script is provided to automate creating a Conda environment and installing the packages from
`requirements.txt` on Windows:

Script: `anaconda-tf-environment.bat` (located in the repository root)

Key points about what the script does and how it behaves:

- Defaults: environment name `be544` and Python `3.10` (you may supply a different name and Python version as positional
  arguments).
- Supported flags: `--no-gpu` (skip attempting to install CUDA/cuDNN), `--force` (remove any existing environment with
  the same name before creating), `--no-pause` (do not pause at the end), `--silent` (suppress console output), and
  `--help`.
    - The script also accepts `--quiet` as an alias for `--silent`.
- The script verifies that `conda` is available on PATH; if not, it exits with a message and non-zero status. Run it
  from an Anaconda Prompt or enable conda in your shell before using the script.
- It attempts a non-fatal `conda update -n base -c defaults conda` early on; the script continues even if the update
  fails.
- Ensures `requirements.txt` exists next to the script; if missing the script exits with an explanatory error.
- Environment creation and removal are handled via `conda create` / `conda env remove`. Subsequent Python/package
  commands are executed inside the environment using `conda run -n "<env>"` so `conda init` is not required.
- If an NVIDIA GPU is detected (presence of `nvidia-smi` on PATH) and `--no-gpu` is not provided, the script attempts to
  install `cudatoolkit` and `cudnn` into the new environment from `conda-forge`.
- The script upgrades `pip` inside the created environment and installs the packages from `requirements.txt` using a
  single pip invocation:
  `conda run -n "<env>" python -m pip install --progress-bar off -r "requirements.txt"`.
- Logging: the script writes a log file named `anaconda-tf-environment.log` next to the script and appends sanitized
  runtime messages. By default the script streams messages to the console and appends the same text to the log; use
  `--silent` to suppress console output while still writing the log.
- By default the script pauses at the end so you can read messages; use `--no-pause` for non-interactive or automated
  runs.

Usage examples (Windows `cmd.exe`):

- Create the default environment named `be544` with Python 3.10 (interactive):

```cmd
anaconda-tf-environment.bat
```

- Create a custom environment `myenv` with Python 3.10 (console shows progress):

```cmd
anaconda-tf-environment.bat myenv 3.10
```

- Create `myenv` but skip GPU/CUDA installation:

```cmd
anaconda-tf-environment.bat myenv 3.10 --no-gpu
```

- Recreate an existing environment (force remove then create):

```cmd
anaconda-tf-environment.bat myenv 3.10 --force
```

- Run non-interactively (do not pause at the end):

```cmd
anaconda-tf-environment.bat myenv 3.10 --no-pause
```

- Run silently (suppress console output, log still written):

```cmd
anaconda-tf-environment.bat --silent
```

If you prefer to set up the environment manually, you can run the following commands in an Anaconda Prompt:

```cmd
conda create -n be544 python=3.10 -y
conda activate be544
pip install -r requirements.txt
```

## Dataset, Extracted Patches, and Code

**Datasets**:

***BACH Dataset : Grand Challenge on Breast Cancer Histology images***:

> The dataset is composed of Hematoxylin and eosin (H&E) stained breast histology microscopy and whole-slide images.
> Challenge participants should evaluate the performance of their method on either/both sets of images.

Challenge Link: https://iciar2018-challenge.grand-challenge.org/Dataset/

> Citation: Polónia, A., Eloy, C., & Aguiar, P. (2019). BACH Dataset : Grand Challenge on Breast Cancer Histology
> images [Data set]. In Medical Image Analysis (Vol. 56, pp. 122–139). Zenodo. https://doi.org/10.5281/zenodo.3632035

_Disclaimer: The datasets are provided for educational purposes only. They are publicly available and can be
accessed from their original links. The author, myself, does not own the datasets._

**Extracted Patches**:

You can use the extracted patches (link below) from the dataset called the "BACH Dataset." These patches are organized
for training, testing, and validation across the three magnification levels and four classes. To download the extracted
patches, you can use the following link:

> https://drive.google.com/drive/folders/1FqSs-xWs-vvcHUl3ojkBbTzZl-JEK9oJ

Each compressed ROI/Tiles file in the previous link will comprise three subfolders: train, val, and test. Within each of
these, there will be four inner subfolders representing the four categories. The naming of the compressed files/folders
pattern is `{ROIs/Tiles}_{level}_{width}_{height}_{width overlap}_{height overlap}_Split`.

For replicability, the dataset's structure and associated metadata are detailed in the table below. The overlap measures
32 in width and height for level 0, 4 for levels 1 and 2. A maximum of 2,000 samples are extracted from each region. The
tolerance, based on the base level, is set to 0.7. The Python snippet `[8] Extract All Tiles with Overlapping.py` is
employed for this purpose.

The updated annotations that include the `Normal` class are available in the `BACH Dataset - Updated Annotations`
folder.

_Disclaimer: The annotations for that class are done manually by me. The annotations are provided for educational
purposes only._

<center>
<table align="center" style="font-size: smaller;margin-left: auto;margin-right: auto;">
    <tr>
        <th></th>
        <th>Level 0 (Base Level)</th>
        <th>Level 1</th>
        <th>Level 2</th>
    </tr>
    <tr>
        <td>ROI Filename</td>
        <td>ROIs_0_256_256_32_32_Split</td>
        <td>ROIs_1_256_256_4_4_Split</td>
        <td>ROIs_2_256_256_4_4_Split</td>
    </tr>
    <tr>
        <td>Tiles Filename</td>
        <td>Tiles_0_256_256_32_32_Split</td>
        <td>Tiles_1_256_256_4_4_Split</td>
        <td>Tiles_2_256_256_4_4_Split</td>
    </tr>
    <tr>
        <td>Shape (Width x Height)</td>
        <td>256x256</td>
        <td>256x256</td>
        <td>256x256</td>
    </tr>
    <tr>
        <td>Overlap (Width x Height)</td>
        <td>32x32</td>
        <td>4x4</td>
        <td>4x4</td>
    </tr>
    <tr>
        <td># Patches Count</td>
        <td>12,000 (3K/Class)</td>
        <td>1,500 (375/Class)</td>
        <td>280 (70/Class)</td>
    </tr>
    <tr>
        <td># Training Count</td>
        <td>8,000 (2K/Class)</td>
        <td>1,100 (275/Class)</td>
        <td>200 (50/Class)</td>
    </tr>
    <tr>
        <td># Validation Count</td>
        <td>2,000 (500/Class)</td>
        <td>200 (50/Class)</td>
        <td>40 (10/Class)</td>
    </tr>
    <tr>
        <td># Testing Count</td>
        <td>2,000 (500/Class)</td>
        <td>200 (50/Class)</td>
        <td>40 (10/Class)</td>
    </tr>
</table>
</center>

**Code**:

All code used in the lectures will be available in this GitHub
repository (https://github.com/HossamBalaha/BE-544-Artificial-Intelligence-Techniques-in-Digital-Histopathology) in
the `Lectures Scripts` folder.

## Copyright and License

No part of this series may be reproduced, distributed, or transmitted in any form or by any means, including
photocopying, recording, or other electronic or mechanical methods, without the prior written permission of the author,
except in the case of brief quotations embodied in critical reviews and certain other noncommercial uses permitted by
copyright law.
For permission requests, contact the author.

The code provided in this series is for educational purposes only and should be used with caution. The author is not
responsible for any misuse of the code provided.

## Citations and Acknowledgments

If you find this series helpful and use it in your research or projects, please consider citing it as:

```bibtex
@software{Balaha_BE_645_Artificial_2024,
  author  = {Balaha, Hossam Magdy},
  month   = jun,
  title   = {{BE 544 Artificial Intelligence (AI) Techniques in Digital Histopathology (Summer 2024)}},
  url     = {https://github.com/HossamBalaha/BE-544-Artificial-Intelligence-Techniques-in-Digital-Histopathology},
  version = {1.06.20},
  year    = {2024}
}

@software{hossam_magdy_balaha_2024_12170422,
  author    = {Hossam Magdy Balaha},
  title     = {{HossamBalaha/BE-544-Artificial-Intelligence-Techniques-in-Digital-Histopathology: v1.06.20}},
  month     = jun,
  year      = 2024,
  publisher = {Zenodo},
  version   = {v1.06.20},
  doi       = {10.5281/zenodo.12192041},
  url       = {https://doi.org/10.5281/zenodo.12192041}
}
```

## Contact

This series is prepared and presented by `Hossam Magdy Balaha` from the University of Louisville's J.B. Speed School of
Engineering.

For any questions or inquiries, please contact me using the contact information available on my CV at the following
link: https://hossambalaha.github.io/
