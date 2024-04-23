<img src="https://github.com/BANDAS-Center/aTrain/blob/main/docs/images/logo.svg" width="300" alt="Logo">

## Tl;dr
aTrain-core: A command line interface for the transcription of audio and video files ([faster-whisper](https://github.com/SYSTRAN/faster-whisper)), including speaker diarization ([pyannote-audio](https://github.com/pyannote/pyannote-audio)). It runs on Windows, MacOS and Debian-based Linux distributions. Supports all the newest models, including Whisper large V3 and distilled large v3 model (English, real time transcription on CPU).

📝 If you use aTrain for a publication, please cite [our paper](https://www.sciencedirect.com/science/article/pii/S2214635024000066?via%3Dihub) as per our [license](https://github.com/JuergenFleiss/atrain_core?tab=License-1-ov-file). 

## Installation ⚙️

**You need to have python >=3.10 installed**  

First you need to create and activate a virtual environment


**Windows**
```
python -m venv venv
.\venv\Scripts\activate
```

**Debian und MacOS**
```
python3 -m venv venv
source venv/bin/activate
```

Then you need to clone our repository into that directory including the extra index. The extra index downloads the necessary libraries for GPU support. 
💡 Note: At the moment, GPU support is only available for Windows and Debian. We are currently working on a solution for MacOS.

**Windows**
```
pip install aTrain_core@git+https://github.com/JuergenFleiss/aTrain-core.git --extra-index-url https://download.pytorch.org/whl/cu118
```
**Debian and MacOS** 
```
pip3 install aTrain_core@git+https://github.com/JuergenFleiss/aTrain-core.git --extra-index-url https://download.pytorch.org/whl/cu118
```
Linux keeps killing collection of torch? Try adding the flag--no-cache-dir

## Usage 📚

Here's a list of all possible model configurations with **default settings in bold**. 

![Model Configurations](docs/model_configurations.png)


If you wish to keep default settings you can simply run: 
```
aTrain_core transcribe /path/to/audio/file.mp3
```
By adapting the following flags you can specify 

```
aTrain_core transcribe /path/to/audio/file.mp3 --model <MODEL> --language <LANGUAGE> --speaker_detection --num_speakers <NUM_SPEAKERS> --device <DEVICE> --compute_type <COMPUTE_TYPE>
```

💡 Good to know: As opposed to the other specifications where the name of model, etc. has to be stated explicitly, the speaker detection is activated simply by adding the flag ```--speaker_detection```

💡 The specified model in the transcription (either default or other) is automatically downloaded into the ```models```folder in the same ```aTrain_core``` root directory where you can find the ```transcriptions```. The ```aTrain_core```directory can be found in your machine's ```Documents``` folder.

If you wish to download all models (separately from the transcription process), simply run: 

```
aTrain_core load
```






## Accessible Transcription of Interviews
aTrain is a tool for automatically transcribing speech recordings utilizing state-of-the-art machine learning models without uploading any data. It was developed by researchers at the Business Analytics and Data Science-Center at the University of Graz and tested by researchers from the Know-Center Graz. 

Big News! The paper introducing aTrain has been published in the Journal of Behavioral and Experimental Finance. Please now cite the published paper if you used aTrain for your research: [Take the aTrain. Introducing an Interface for the Accessible Transcription of Interviews.](https://www.sciencedirect.com/science/article/pii/S2214635024000066)

Windows (10 and 11) users can install aTrain via the Microsoft app store ([Link](https://apps.microsoft.com/store/detail/atrain/9N15Q44SZNS2)) or by downloading the installer from the BANDAS-Center Website ([Link](https://business-analytics.uni-graz.at/de/forschen/atrain/download/)).

For Linux, follow the [instructions](https://github.com/JuergenFleiss/aTrain/wiki/Linux-Support-(in-progress)) in our Wiki.

aTrain offers the following benefits:
\
\
**Fast and accurate 🚀**
\
aTrain provides a user friendly access to the [faster-whisper](https://github.com/guillaumekln/faster-whisper) implementation of OpenAI’s [Whisper model](https://github.com/openai/whisper), ensuring best in class transcription quality (see [Wollin-Geiring et al. 2023](https://www.static.tu.berlin/fileadmin/www/10005401/Publikationen_sos/Wollin-Giering_et_al_2023_Automatic_transcription.pdf)) paired with higher speeds on your local computer. Transcription when selecting the highest-quality model takes only around three times the audio length on current mobile CPUs typically found in middle-class business notebooks (e.g., Core i5 12th Gen, Ryzen Series 6000).
\
\
**Speaker detection 🗣️**
\
aTrain has a speaker detection mode based on [pyannote.audio](https://github.com/pyannote/pyannote-audio) and can analyze each text segment to determine which speaker it belongs to.
\
\
**Privacy Preservation and GDPR compliance 🔒**
\
aTrain processes the provided speech recordings completely offline on your own device and does not send recordings or transcriptions to the internet. This helps researchers to maintain data privacy requirements arising from ethical guidelines or to comply with legal requirements such as the GDRP.
\
\
**Multi-language support 🌍**
\
aTrain can process speech recordings in any of the following 99 languages: Afrikaans, Albanian, Amharic, Arabic, Armenian, Assamese, Azerbaijani, Basque, Bashkir, Belarusian, Bengali, Bosnian, Breton, Bulgarian, Cantonese, Catalan, Croatian, Czech, Danish, Dutch, English, Estonian, Faroese, Finnish, French, Galician, Georgian, German, Greek, Gujarati, Haitian Creole, Hausa, Hawaiian, Hebrew, Hindi, Hungarian, Icelandic, Igbo, Indonesian, Irish, Italian, Japanese, Javanese, Kannada, Kazakh, Khmer, Kinyarwanda, Korean, Kurdish, Kyrgyz, Lao, Latin, Latvian, Lingala, Lithuanian, Lao, Luxembourgish, Macedonian, Malagasy, Malay, Malayalam, Maltese, Maori, Marathi, Mongolian, Myanmar, Nepali, Norwegian, Nynorsk, Occitan, Oriya, Oromo, Pashto, Persian, Polish, Portuguese, Punjabi, Quechua, Romanian, Russian, Sanskrit, Scots Gaelic, Serbian, Sindhi, Sinhala, Slovak, Slovenian, Somali, Sotho, Spanish, Sundanese, Swahili, Swedish, Tajik, Tamil, Tatar, Telugu, Thai, Tibetan, Tigrinya, Tongan, Turkish, Turkmen, Ukrainian, Urdu, Uyghur, Uzbek, Vietnamese, Welsh, Xhosa, Yiddish, Yoruba, Zulu.
\
\
**MAXQDA and ATLAS.ti compatible output 📄**
\
aTrain provides transcription files that are seamlessly importable into the most popular tools for qualitative analysis, ATLAS.ti and MAXQDA. This allows you to directly play audio for the corresponding text segment by clicking on its timestamp. Go to the [tutorial](https://github.com/BANDAS-Center/aTrain/wiki/Tutorials).
\
\
**Nvidia GPU support 🖥️**
\
aTrain can either run on the CPU or an NVIDIA GPU (CUDA toolkit installation required). A [CUDA-enabled NVIDIA GPU](https://developer.nvidia.com/cuda-gpus) significantly improves the speed of transcriptions and speaker detection, reducing transcription time to 20% of audio length on current entry-level gaming notebooks.

| Screenshot 1 | Screenshot 2 |
| --- | --- |
| ![Screenshot1](docs/images/screenshot_1.webp) | ![Screenshot2](docs/images/screenshot_2.webp) |

## Benchmarks
For testing the processing time of aTrain-core we transcribed an audiobook ("[The Snow Queen](https://ia802608.us.archive.org/33/items/andersens_fairytales_librivox/fairytales_06_andersen.mp3)" from Hans Christian Andersen with a duration of 1 hour, 13 minutes, and 38 seconds) with three different computers (see table 1). The figure below shows the processing time of each transcription relative to the length of the speech recording. In this relative processing time (RPT), a transcription is considered ’real time’ when the recording length and the processing time are equal. Subsequently, faster transcriptions lead to an RPT below 1 and slower transcriptions to an RPT time above 1.

| Benchmark results | Used hardware |
| --- | --- |
| ![Benchmark](docs/images/benchmark.webp) | ![Hardware](docs/images/hardware.webp) |

## System requirements
We support Windows, Debian and MacOS. GPU transcription with NVIDIA CUDA GPUs is supported in Windows and Debian. 


