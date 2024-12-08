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
   python synthesize.py 'inferencer.input_text="TEXT"' inferencer.save_path=SAVE_PATH
   ```
   where `SAVE_PATH` is a path to save synthesized audio.

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
   python synthesize.py input_text_dir=DIR_PATH save_path=SAVE_PATH
   ```
   where `DIR_PATH` is directory with text and `SAVE_PATH` is a path to save synthesized audio.

2) If you want to resynthesize audio from .wav files, your directory with utterences should has the following format:
   ```
   NameOfTheDirectoryWithUtterances
   └── audios
        ├── UtteranceID1.wav
        ├── UtteranceID2.wav
        .
        .
        .
        └── UtteranceIDn.wav
   ```
   Run the following command:
   ```bash
   python synthesize.py input_audio_dir=DIR_PATH save_path=SAVE_PATH
   ```
   where `DIR_PATH` is directory with utterences and `SAVE_PATH` is a path to save resynthesized audio.

### Training

To reproduce this model, run the following command:

   ```bash
   python train.py \
      writer.run_name=RUN_NAME \


   python train.py \
      datasets.train.max_audio_length=32768 \
      datasets.val.max_audio_length=32768 \
      trainer.n_epochs=15 \
      +trainer.from_pretrained="saved/RUN_NAME/model_best.pth"
   ```

## Final results

Audios presented in order <ground truth audio/predict>

- `Mihajlo Pupin was a founding member of National Advisory Committee for Aeronautics (NACA) on 3 March 1915, which later became NASA, and he participated in the founding ofAmerican Mathematical Society and American Physical Society.`


https://github.com/user-attachments/assets/ad5aabcc-9ca6-4297-a16b-709605d51709
https://github.com/user-attachments/assets/e00ed576-5880-4569-9740-3740ee0fd524

- `Leonard Bernstein was an American conductor, composer, pianist, music educator, author, and humanitarian. Considered to be one of the most important conductors of his time, he was the first American-born conductor to receive international acclaim.`

https://github.com/user-attachments/assets/ca11d5ff-fd96-420d-a04e-c9997b568122
https://github.com/user-attachments/assets/86737342-d41a-402f-8930-ac6ed3ac5403

- `Lev Termen, better known as Leon Theremin was a Russian inventor, most famous for his invention of the theremin, one of the first electronic musical instruments and the first to be mass-produced.`

https://github.com/user-attachments/assets/d7e084c5-0af8-4aed-95b1-5f25f8ec9d24
https://github.com/user-attachments/assets/f59ec89c-7ac1-40d7-a484-93035e5c95da

- `Deep Learning in Audio course at HSE University offers an exciting and challenging exploration of cutting-edge techniques in audio processing, from speech recognition to music analysis. With complex homeworks that push students to apply theory to real-world problems, it provides a hands-on, rigorous learning experience that is both demanding and rewarding.`

https://github.com/user-attachments/assets/e92b4589-b9da-4ba7-ba9b-2f0e3c2b3fa6
https://github.com/user-attachments/assets/a386eaf2-2867-4caa-b445-4d5ea2fee6f1

- `Dmitri Shostakovich was a Soviet-era Russian composer and pianist who became internationally known after the premiere of his First Symphony in 1926 and thereafter was regarded as a major composer.`

https://github.com/user-attachments/assets/bd940f23-0e8c-4942-b11d-308f06988106
https://github.com/user-attachments/assets/9a8c8dbf-ac04-40e2-86c2-4e2ef7d0c8dd

 `WV-MOS=2.78`.

## Credits

This repository is based on a [PyTorch Project Template](https://github.com/Blinorot/pytorch_project_template).

## License

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
