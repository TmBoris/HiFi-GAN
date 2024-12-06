<h1 align="center">HiFi-GAN</h1>

<p align="center">
  <a href="#about">About</a> •
  <a href="#installation">Installation</a> •
  <a href="#how-to-use">How To Use</a> •
   <a href="#final-results">Final results</a> •
  <a href="#credits">Credits</a> •
  <a href="#license">License</a>
</p>

## About

Implementation is based on the [HiFiGAN](https://arxiv.org/pdf/2010.05646) paper.

See the task assignment [here](https://github.com/markovka17/dla/tree/2024/hw3_nv).

You can find implementation details and audio analysis in my [WandB report](https://api.wandb.ai/links/bspanfilov/r82qwddg).

## Installation

Follow these steps to install the project:

0. (Optional) Create and activate new environment
   using [`conda`](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html).

   ```bash
   # create env
   conda create -n hifi python=3.11

   # activate env
   conda activate hifi
   ```

1. Install all required packages.

   ```bash
   pip install -r requirements.txt
   ```
2. Download model checkpoint.

   ```bash
   python download_weights.py
   ```

## How To Use

### Inference

1) If you only want to synthesize one text/phrase and save it, run the following command:

   ```bash
   python synthesize.py 'text="YOUR_TEXT"' save_path=SAVE_PATH
   ```
   where `SAVE_PATH` is a path to save synthesize audio. Please be careful in quotes.

2) If you want to synthesize audio from text files, your directory with text should has the following format:
   ```
   NameOfTheDirectoryWithUtterances
   └── transcriptions
        ├── UtteranceID1.txt
        ├── UtteranceID2.txt
        .
        .
        .
        └── UtteranceIDn.txt
   ```
   Run the following command:
   ```bash
   python synthesize.py dir_path=DIR_PATH save_path=SAVE_PATH
   ```
   where `DIR_PATH` is directory with text and `SAVE_PATH` is a path to save synthesize audio.

### Training

To reproduce this model, run the following command:

   ```bash
   python train.py
   ```

## Final results

Audios presented in order <ground truth audio/predict>

- `Mihajlo Pupin was a founding member of National Advisory Committee for Aeronautics (NACA) on 3 March 1915, which later became NASA, and he participated in the founding ofAmerican Mathematical Society and American Physical Society.`

<audio controls src="data/saved/resynth_grading_with_ft/gt_audio/pupin.wav" title="gt_audio"></audio>
<audio controls src="data/saved/resynth_grading_with_ft/pr_audio/pupin.wav" title="predict"></audio>


- `Leonard Bernstein was an American conductor, composer, pianist, music educator, author, and humanitarian. Considered to be one of the most important conductors of his time, he was the first American-born conductor to receive international acclaim.`

<audio controls src="data/saved/resynth_grading_with_ft/gt_audio/bernstein.wav" title="gt_audio"></audio>
<audio controls src="data/saved/resynth_grading_with_ft/pr_audio/bernstein.wav" title="predict"></audio>

- `Lev Termen, better known as Leon Theremin was a Russian inventor, most famous for his invention of the theremin, one of the first electronic musical instruments and the first to be mass-produced.`

<audio controls src="data/saved/resynth_grading_with_ft/gt_audio/theremin.wav" title="gt_audio"></audio>
<audio controls src="data/saved/resynth_grading_with_ft/pr_audio/theremin.wav" title="predict"></audio>

- `Deep Learning in Audio course at HSE University offers an exciting and challenging exploration of cutting-edge techniques in audio processing, from speech recognition to music analysis. With complex homeworks that push students to apply theory to real-world problems, it provides a hands-on, rigorous learning experience that is both demanding and rewarding.`

<audio controls src="data/saved/resynth_grading_with_ft/gt_audio/dla.wav" title="gt_audio"></audio>
<audio controls src="data/saved/resynth_grading_with_ft/pr_audio/dla.wav" title="predict"></audio>

- `Dmitri Shostakovich was a Soviet-era Russian composer and pianist who became internationally known after the premiere of his First Symphony in 1926 and thereafter was regarded as a major composer.`

<audio controls src="data/saved/resynth_grading_with_ft/gt_audio/shostakovich.wav" title="gt_audio"></audio>
<audio controls src="data/saved/resynth_grading_with_ft/pr_audio/shostakovich.wav" title="predict"></audio>


 `WV-MOS=2.78`.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
