#!/usr/bin/env python
from __future__ import absolute_import, division, print_function

# Make sure we can import stuff from util/
# This script needs to be run from the root of the DeepSpeech repository
import os
import sys

sys.path.insert(1, os.path.join(sys.path[0], '..'))

import csv
import re
from util.audio import audiofile_to_input_vector

from glob import glob
from os import path

# language = 'english'
language = 'arabic'
# data_path = '/data/abbas/repos/DeepSpeech/'
# data_path = '/Users/aali/Data/Professional/ADFG/Projects/Speech/DeepSpeech/'
data_path = ''

# corpus_data = '/data/abbas/corpora/english_speech_corpus/'
# corpus_data = '/Users/aali/Data/Professional/ADFG/Projects/Speech/Datasets/data/'

global corpus_dir
# if('english' in language):
corpus_dir = '/ain3/ml3_data/abbas/data/' + language + '/'
# elif('arabic' in language):
#     corpus_data = '/data/abbas/data/'

features_dir = data_path + 'features/' + language + '/deepspeech_2/'

if ('english' in language):
    transcript_features_ratio = 1.1
    transcript_len = 230

    # corpus_dir = corpus_data + language + '/'
    # datasets_train_test = ['vctk','tedlium3','voxforge']
    # datasets_train_test = ['librispeech', 'common_voice', 'voxforge', 'tedlium1', 'tidigits', 'tedlium2', 'timit', 'tedlium3', 'vctk', 'l2_arctic']
    datasets_train_test = ['librispeech', 'voxforge', 'tidigits', 'vctk', 'l2_arctic', 'tedlium3']
    # datasets_train_test = ['librispeech', 'common_voice', 'voxforge', 'tidigts', 'tedlium2', 'timit', 'tedlium3', 'vctk', 'l2_arctic']
    # datasets_train_test = ['common_voice', 'tidigts', 'timit', 'vctk', 'l2_arctic']
    # datasets_train_test = ['vctk']
    # datasets_train_test = ['librispeech','tedlium1','voxforge','tedlium2','tidigts','tedlium3','timit','vctk']
    # datasets_train_test = ['librispeech','tedlium1','tidigts','tedlium2','timit']
    datasets_dev = ['librispeech', 'voxforge', 'tedlium1', 'tidigits', 'tedlium2', 'tedlium3', 'vctk', 'l2_arctic']
    # datasets_dev = ['common_voice']
else:
    transcript_features_ratio = 1.1
    transcript_len = 200

    # corpus_dir = '/data/abbas/data/' + language + '/'
    # datasets_train_test = ['kacst','ksu','al_jazeera','isolated_words','mgb2','aldiri']
    # datasets_dev = ['kacst','ksu','al_jazeera','isolated_words','mgb2','aldiri']
    datasets_train_test = ['kacst','ksu','mgb2']
    datasets_dev = ['kacst','ksu','mgb2']
    # datasets_train_test = ['kacst','ksu','al_jazeera','isolated_words','ainfinity','mgb2','aldiri']
    # datasets_dev = ['kacst','ksu','al_jazeera','isolated_words','ainfinity','mgb2','aldiri']
    # datasets_train_test = ['mgb2']
    # datasets_dev = ['mgb2']

FIELDNAMES = ['wav_filename', 'wav_filesize', 'transcript']
# SAMPLE_RATE = 16000
# MAX_SECS = 10
# ARCHIVE_DIR_NAME = 'cv_corpus_v1'
# ARCHIVE_NAME = ARCHIVE_DIR_NAME + '.tar.gz'
# ARCHIVE_URL = 'https://s3.us-east-2.amazonaws.com/common-voice-data-download/' + ARCHIVE_NAME

# SIMPLE_BAR = ['Progress ', progressbar.Bar(), ' ', progressbar.Percentage(), ' completed']

# def _download_and_preprocess_data(target_dir):
#     # Making path absolute
#     target_dir = path.abspath(target_dir)
#     # Conditionally download data
#     archive_path = _maybe_download(ARCHIVE_NAME, target_dir, ARCHIVE_URL)
#     # Conditionally extract common voice data
#     _maybe_extract(target_dir, ARCHIVE_DIR_NAME, archive_path)
#     # Conditionally convert common voice CSV files and mp3 data to DeepSpeech CSVs and wav
#     _maybe_convert_sets(target_dir, ARCHIVE_DIR_NAME)
#
# def _maybe_download(archive_name, target_dir, archive_url):
#     # If archive file does not exist, download it...
#     archive_path = path.join(target_dir, archive_name)
#     if not path.exists(archive_path):
#         print('No archive "%s" - downloading...' % archive_path)
#         req = requests.get(archive_url, stream=True)
#         total_size = int(req.headers.get('content-length', 0))
#         done = 0
#         with open(archive_path, 'wb') as f:
#             bar = progressbar.ProgressBar(max_value=total_size, widgets=SIMPLE_BAR)
#             for data in req.iter_content(1024*1024):
#                 done += len(data)
#                 f.write(data)
#                 bar.update(done)
#     else:
#         print('Found archive "%s" - not downloading.' % archive_path)
#     return archive_path
#
# def _maybe_extract(target_dir, extracted_data, archive_path):
#     # If target_dir/extracted_data does not exist, extract archive in target_dir
#     extracted_path = path.join(target_dir, extracted_data)
#     if not path.exists(extracted_path):
#         print('No directory "%s" - extracting archive...' % archive_path)
#         with tarfile.open(archive_path) as tar:
#             tar.extractall(target_dir)
#     else:
#         print('Found directory "%s" - not extracting it from archive.' % archive_path)
#
# def _maybe_convert_sets(target_dir, extracted_data):
#     extracted_dir = path.join(target_dir, extracted_data)
#     for source_csv in glob(path.join(extracted_dir, '*.csv')):
#         _maybe_convert_set(extracted_dir, source_csv, path.join(target_dir, os.path.split(source_csv)[-1]))

punctuationList = [".", "ـ", ";", ":", "!", "?", "/", "\\", ",", "#", "@", "*", "$", "&", ")", "(", "\"", "^", "`", "´",
                   "[", "]", "¬", "<", "{", "£", "؟", "}", "-", "­", "،", "٠", "ے", "‏", ">", "~", "©", "®", "º", "+",
                   "\xa0", "\u2028", "\ufeff", "ß", "ŕ", 'ۢ', 'ۭ', '۠', 'ۜ', '۬', '۪', 'ۣ', '۫', '۪', 'ۨ', 'ٔ', 'ٰ',
                   '۟', 'ۥ', 'ۦ', "“", "”", "ﬁ", "▪", "�", "ﬂ", "ø", "½", "‑", "ѕ", "„", "", "о", "‟", "ﬀ", "и", "➤",
                   "с", "к", "±", "æ", "", "♦", "─", "‹", "№", "а", "¢", "в", "œ", "』", "т", "¼", "у", "ο", "ß", "¾", "п",
                   "", "⎯", "н", "¨", "µ", "ʼ", "β", "ʔ", "―", "д", "π", "", "ч", "μ", "❍", "р", "ð", "ą",
                   "³", "л", "ﬃ", "α", "з", "σ", "г", "м", "τ", "λ", "ī", "й", "ς", "‡", "☛", "ь", "ρ", "ж", "≥", "ы",
                   "þ", "合", "‐", "″", "▧", "ђ", "ω", "γ", "¹", "щ", "☺", "ν", "˚", "抰", "出", "‰", "", "į",
                   ";", "ό", "ш", "θ", "я", "đ", "❑", "✔", "ї", "ю", "δ", "б", "⁄", "̹", "ґ", "∙", "☆", "̹", "ґ", "◗",
                   "ˇ", "¸", "ǫ", "′", "ẕ", "っ", "❁", "ε", "ə", "抦", "◊", "і", "˝", "ٱ", "・", "ι", "",
                   "日", "磘", "ɑ", "", "ơ", "※", "ż", "ב", "ד", "篆", "υ", "钱", "剥", "ו", "喂", "♥", "э", "❋", "者", "✓",
                   "ª", "♣", "切", "ά", "љ", "ц", "只", "≤", "▲", "ф", "赤", "✦", "ờ", "ג", "ן", "ṛ", "館", "",
                   "", "訪", "話", "➢", "嶺", "ș", "φ", "", "ふ", "", "王", "", "म", "止", "χ", "ร", "ợ", "ℓ", "塗", "実",
                   "危", "κ", "間", "吴", "￥", "回", "便", "", "ƒ", "¬", "張", "➙", "磗", "惧",
                   "漫", "飛", "詰", "幾", "ξ", "連", "手", "憶", "物", "吹", "", "頑", "ं", "谋", "足", "⁯", "ћ", "人", "˛", "מ",
                   "ů", "自", "電", "ề", "", "考", "持", "", "頂", "", "έ", "บ", "प", "ἄ", "❚",
                   "喜", "煮", "体", "र", "伝", "捌", "取", "⇒", "ई", "♀", "折", "携", "", "思", "ų", "◇", "", "ɪ", "", "宝",
                   "ị", "", "見", "元", "ה", "谷", "寅", "気", "勿", "負", "", "", "ぐ", "ū",
                   "の", "渋", "ċ", "", "", "ब", "玉", "比", "画", "ค", "上", "唸", "搾", "會", "ť", "ा", "引", "ϋ", "倒", "",
                   "ั", "無", "『", "助", "是", "①", "є", "", "य", "帯", "∑", "", "分", "三", "η", "滑",
                   "‚", "ु", "﴾", "﴿", "ּ", "́", "ַ", "ָ", "ַ", "̂", "–", "؛", "…", "‘", "’", "‍", "\u200d", "«", "»",
                   "•", "=", "١", "٢", "٣", "٤", "٥", "٦", "٧", "٨", "٩", "ō", "ā"]

apos_re = re.compile(r"(?<!\w)\'|\'(?!\w)")


def _maybe_convert_set(source_dir, target_dir, mode, datasets):
    rows = []

    remove_alphabets = set('۱١٢۳٣٤٥٦٧۷۸٨٩۹٠۰0123456789٪éàçèáâïóöúﺠپچﭽ')

    for dataset in datasets:
        for subdir, dirs, files in os.walk(source_dir + '/' + dataset + '/' + mode):
            # for audio_filename in sorted(glob.iglob(corpus_dir + "/" + '/**/*.' + ext, recursive=True)):
            for file in files:
                if file.endswith('.txt'):
                    filepath = path.abspath(subdir + '/' + file).split('.')[:-1][0]
                    if path.exists(filepath + '.txt') and path.exists(filepath + '.wav'):
                        with open(filepath + '.txt', 'r') as readfile:
                            for transcript in readfile.readlines():
                                features_len = audiofile_to_input_vector(filepath + '.wav', numcep=26, numcontext=9, compute_len=True, model='deepspeech_2')
                                if(features_len > 100 and len(transcript) > 2):
                                    if ('english' in language):
                                        if ('tedlium' in dataset):
                                            if (len(transcript) >= 7 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= (len(transcript) * transcript_features_ratio)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                        elif ('tidigits' in dataset):
                                            if (len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= len(transcript)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                        elif ('voxforge' in dataset):
                                            if (len(transcript) >= 2 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= len(transcript)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                        elif ('vctk' in dataset):
                                            if (len(transcript) >= 2 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= (len(transcript) * transcript_features_ratio)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                        elif ('common_voice' in dataset):
                                            if (len(transcript) >= 8 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= (len(transcript) * transcript_features_ratio)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                        elif (len(transcript) >= 5 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                            rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                            # if (features_len <= len(transcript)):
                                            #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                    else:
                                        transcript = transcript.replace('آ', 'آ')
                                        transcript = transcript.replace('ﻻ', 'لا')
                                        transcript = transcript.replace('ﻵ', 'لآ')
                                        transcript = transcript.replace('ﻷ', 'لأ')
                                        transcript = transcript.replace('ﻹ', 'لإ')
                                        transcript = transcript.replace('ﺇ', 'إ')
                                        transcript = transcript.replace('ک', 'ك')
                                        transcript = transcript.replace('ی', 'ى')
                                        transcript = transcript.replace('‎‌', ' ')
                                        transcript = transcript.replace('‎', ' ')

                                        # remove diacritics
                                        transcript = transcript.replace('ً', '')
                                        transcript = transcript.replace('ٍ', '')
                                        transcript = transcript.replace('ٌ', '')
                                        transcript = transcript.replace('ْ', '')

                                        # normalization
                                        transcript = transcript.replace('َ', '')
                                        transcript = transcript.replace('ِ', '')
                                        transcript = transcript.replace('ُ', '')
                                        transcript = transcript.replace('ّ', '')
                                        transcript = transcript.replace('ؤ', 'ؤ')
                                        transcript = transcript.replace('ئ', 'ىٔ')
                                        transcript = transcript.replace('أ', 'أ')

                                        if not any((c in remove_alphabets) for c in transcript):
                                            if ('ksu' in dataset):
                                                if (len(transcript) >= 2 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                    rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                                # if (features_len <= (len(transcript) * transcript_features_ratio)):
                                                #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))
                                            else:
                                                if (len(transcript) >= 2 and len(transcript) <= transcript_len and features_len > (len(transcript) * transcript_features_ratio)):
                                                    rows.append((filepath + '.wav', path.getsize(filepath + '.wav'), transcript))
                                                # if (features_len <= (len(transcript) * transcript_features_ratio)):
                                                #     print('Error: Audio file {} is too short for transcription.'.format(filepath + '.wav') + " -- " + str(features_len) + " < " + str(len(transcript)))

    # # if path.exists(target_dir + '/dataset.csv'):
    # samples = []
    # with open(source_csv) as source_csv_file:
    #     reader = csv.DictReader(source_csv_file)
    #     for row in reader:
    #         samples.append((row['filename'], row['text']))
    #
    # # Mutable counters for the concurrent embedded routine
    # counter = { 'all': 0, 'too_short': 0, 'too_long': 0 }
    # lock = RLock()
    # num_samples = len(samples)
    # rows = []
    #
    # def one_sample(sample):
    #     mp3_filename = path.join(*(sample[0].split('/')))
    #     mp3_filename = path.join(extracted_dir, mp3_filename)
    #     # Storing wav files next to the mp3 ones - just with a different suffix
    #     wav_filename = path.splitext(mp3_filename)[0] + ".wav"
    #     _maybe_convert_wav(mp3_filename, wav_filename)
    #     frames = int(subprocess.check_output(['soxi', '-s', wav_filename], stderr=subprocess.STDOUT))
    #     file_size = path.getsize(wav_filename)
    #     with lock:
    #         if int(frames/SAMPLE_RATE*1000/10/2) < len(str(sample[1])):
    #             # Excluding samples that are too short to fit the transcript
    #             counter['too_short'] += 1
    #         elif frames/SAMPLE_RATE > MAX_SECS:
    #             # Excluding very long samples to keep a reasonable batch-size
    #             counter['too_long'] += 1
    #         else:
    #             # This one is good - keep it for the target CSV
    #             rows.append((wav_filename, file_size, sample[1]))
    #         counter['all'] += 1
    #
    # print('Importing mp3 files...')
    # pool = Pool(cpu_count())
    # bar = progressbar.ProgressBar(max_value=num_samples, widgets=SIMPLE_BAR)
    # for i, _ in enumerate(pool.imap_unordered(one_sample, samples), start=1):
    #     bar.update(i)
    # bar.update(num_samples)
    # pool.close()
    # pool.join()
    #
    # print('Writing "%s"...' % target_csv)
    # if ('english' in language):
    # dict_ = {}
    rows.sort(key=lambda item: int(item[1]))
    with open(target_dir + '/' + mode + '.csv', 'w') as target_csv_file:
        writer = csv.DictWriter(target_csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        # bar = progressbar.ProgressBar(max_value=len(rows), widgets=SIMPLE_BAR)

        for filename, file_size, transcript in rows:
            include = True

            transcript = transcript.lower()

            transcript = apos_re.sub('', transcript)  # remove qoutes

            transcript = transcript.replace('-', ' ')

            # transcript = ''.join(['-'.join(c for c in s if c not in punctuationList) for s in transcript])
            transcript = transcript.replace('  ', ' ').replace('\r', ' ').replace('\n', ' ').replace('\t', ' ').replace(
                '_', ' ').lower().strip()

            for c in transcript:
                if (c in punctuationList):
                    include = False
                    break

            if (include):
                writer.writerow({'wav_filename': filename, 'wav_filesize': file_size, 'transcript': transcript})

                # for alpha in transcript:
                #     if (not alpha in dict_):
                #         dict_[alpha] = 1
                #     else:
                #         dict_[alpha] += 1

    # with open(target_dir + '/' + mode + '.txt', 'w') as language_model_file:
    #     for k in sorted(dict_, key=dict_.get, reverse=True):
    #         if (len(k) > 0):
    #             language_model_file.write(k + '\n')  # '\t' + str(military_dict[k]) +
    # else:
    #     dataFrame = pd.DataFrame(rows, columns=['wav_filename', 'wav_filesize', 'transcript'], dtype=int)
    #     dataFrame.to_csv(target_dir + '/' + mode + '.csv', sep=',', header=True, index=False)


#
# def _maybe_convert_wav(mp3_filename, wav_filename):
#     if not path.exists(wav_filename):
#         transformer = Transformer()
#         transformer.convert(samplerate=SAMPLE_RATE)
#         transformer.build(mp3_filename, wav_filename)

if __name__ == "__main__":
    # sys.argv[1]
    _maybe_convert_set(corpus_dir, features_dir, 'train', datasets_train_test)
    _maybe_convert_set(corpus_dir, features_dir, 'test', datasets_train_test)
    _maybe_convert_set(corpus_dir, features_dir, 'dev', datasets_dev)
