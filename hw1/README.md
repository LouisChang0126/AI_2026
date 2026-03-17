# Dataset documentation
There are two datasets: <b>dataset1</b> and <b>dataset2</b>, the first one is the small one, and the second is bigger.

Each dataset contains 5 folders: Drum_solo, Piano_Solo, Violin_Solo, Acoustic_Guitar_Solo, Electric_Guitar_Solo, each folder contains the sound of the instrument, each sound is 10 seconds.

<b>Data type:</b> wav

<b>External source:</b> YouTube

<b>Amount:</b> 437 / 819 of wav file in two datasets, <b>dataset2 is nearly 2 times larger than dataset1</b>.

<b>Data collection:</b> I use 2 Python codes to collect the dataset, the first one is download_youtube_webm.py, which can web crawl from YouTube and download 10 webm files with a specific keyword. After manual checking that the file is actually "Solo" by that instrument, run the second Python file: webm_to_wav.py. It will convert the webm file to wav, cut each file to 10 / 20 10-second pieces, and save the pieces into the dataset folder.(Need to download ffmpeg for data type converting)

![Data collection pipeline](docs\data_collect_pipeline.png)

<b>Example:</b> /dataset1/Drum_Solo/ Benny Greb Drum Solo - Drumeo_part2.wav, /dataset2/Electric_Guitar_Solo/Van Halen - Beat It Solo Cover_part5.wav are two of the data.
