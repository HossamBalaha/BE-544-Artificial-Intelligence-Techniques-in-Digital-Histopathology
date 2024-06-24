# BE 544 Artificial Intelligence (AI) Techniques in Digital Histopathology (Summer 2024)

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

**Full Playlist**:
Link: https://www.youtube.com/playlist?list=PLVrN2LRb7eT3_la39bWC0EP-IW5jNjQ-w

**Videos**:

1. [BE544 Lecture 01 - Intro to Histopathology](https://youtu.be/e6RCziIUaB8)
2. [BE544 Lecture 02 - Intro to Histopathology (Contd.)](https://youtu.be/HcG7DQJgFvQ)
3. [BE544 Lecture 03 - Intro to QuPath](https://youtu.be/m2rRXoqZWOg)
4. [BE544 Lecture 04 - Intro to Aperio ImageScope](https://youtu.be/1p0fDFCv34s)
5. [BE544 Lecture 05 - Working with Annotations and Extracting Tiles](https://youtu.be/GDrhFgeukt8)
6. [BE544 Lecture 06 - Working with Annotations and Extracting Tiles (Contd.)](https://youtu.be/TX3AUxNZVLU)
7. [BE544 Lecture 07 - Classification using Convolutional Neural Network](https://youtu.be/lZSJqs9xrJM)
8. [BE544 Lecture 08 - Classification using Convolutional Neural Network (Contd.)](https://youtu.be/erSsRc7BIQM)
9. [BE544 Lecture 09 - Classification using Convolutional Neural Network (Contd.)](https://youtu.be/KBZyldoAM48)

... and more to come!

## Programming Language and Libraries

The programming language used in this series is `Python`, and the primary libraries employed are:

1. `OpenSlide` - A C library that provides a simple interface to read whole-slide images (also known as virtual slides).
2. `OpenCV` - An open-source computer vision and machine learning software library.
3. `NumPy` - A fundamental package for scientific computing with Python.
4. `Matplotlib` - A comprehensive library for creating static, animated, and interactive visualizations in Python.
5. `Scikit-learn` - A simple and efficient tool for data mining and data analysis built on NumPy, SciPy, and Matplotlib.
6. `TensorFlow` - An end-to-end open-source platform for machine learning.
7. `Keras` - An open-source neural network library written in Python.
8. `Split Folders` - A simple library to split folders into training, validation, and testing directories.
9. `Keras Tuner` - A library to perform hyperparameter tuning for Keras models.

The packages versions of the libraries used in this series are:

```text
cv2==4.9.0
numpy==1.26.4
matplotlib==3.8.3
scikit-learn==1.4.1.post1
tensorflow==2.10.1
keras==2.10.0
splitfolders==0.5.1
keras_tuner==1.4.7
```

To install the required libraries, you can use the following PIP commands:

```
pip install opencv-python==4.9.*
pip install opencv-contrib-python==4.9.*
pip install numpy==1.26.4
pip install matplotlib==3.8.3
pip install scikit-learn==1.4.1.post1
pip install tensorflow==2.10.1
pip install keras==2.10.0
pip install split-folders==0.5.1
pip install keras-tuner==1.4.7
```

_Disclaimer: The versions of the libraries may change based on updates and releases. However, the code should work
with the latest versions. Please note that the code has been tested on `Python 3.9.17` and the specified library
versions on a `Windows 11` machine. It has not been tested on other operating systems or other versions of Python and
the libraries._

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
