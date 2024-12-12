import os
import shutil

# 元データセットのルートディレクトリ
source_dir = "data/datasets/jvs_ver1"
# 保存先ディレクトリ
output_dir = "data/cache/jvs"

# 保存先ディレクトリを作成（存在しない場合）
os.makedirs(output_dir, exist_ok=True)

# 各スピーカー（jvs001, jvs002, ...）を処理
for speaker in os.listdir(source_dir):
    speaker_path = os.path.join(source_dir, speaker)
    if not os.path.isdir(speaker_path):
        continue

    # 非並列セクション
    nonpara_path = os.path.join(speaker_path, "nonpara30")
    if os.path.exists(nonpara_path):
        wav_dir = os.path.join(nonpara_path, "wav24kHz16bit")
        lab_dir = os.path.join(nonpara_path, "lab/mon")
        if os.path.exists(wav_dir) and os.path.exists(lab_dir):
            for wav_file in os.listdir(wav_dir):
                if wav_file.endswith(".wav"):
                    base_name = os.path.splitext(wav_file)[0]
                    lab_file = f"{base_name}.lab"

                    # 元ファイルのパス
                    wav_path = os.path.join(wav_dir, wav_file)
                    lab_path = os.path.join(lab_dir, lab_file)

                    # 出力ファイル名
                    output_wav = f"{speaker}_nonpara30_{wav_file}"
                    output_lab = f"{speaker}_nonpara30_{lab_file}"

                    # ファイルコピー
                    shutil.copy(wav_path, os.path.join(output_dir, output_wav))
                    if os.path.exists(lab_path):
                        shutil.copy(lab_path, os.path.join(output_dir, output_lab))
    para_path = os.path.join(speaker_path, "parallel100")
    if os.path.exists(nonpara_path):
        wav_dir = os.path.join(para_path, "wav24kHz16bit")
        lab_dir = os.path.join(para_path, "lab/mon")
        if os.path.exists(wav_dir) and os.path.exists(lab_dir):
            for wav_file in os.listdir(wav_dir):
                if wav_file.endswith(".wav"):
                    base_name = os.path.splitext(wav_file)[0]
                    lab_file = f"{base_name}.lab"

                    # 元ファイルのパス
                    wav_path = os.path.join(wav_dir, wav_file)
                    lab_path = os.path.join(lab_dir, lab_file)

                    # 出力ファイル名
                    output_wav = f"{speaker}_parallel100_{wav_file}"
                    output_lab = f"{speaker}_parallel100_{lab_file}"

                    # ファイルコピー
                    shutil.copy(wav_path, os.path.join(output_dir, output_wav))
                    if os.path.exists(lab_path):
                        shutil.copy(lab_path, os.path.join(output_dir, output_lab))


print("データの変換が完了しました！")
