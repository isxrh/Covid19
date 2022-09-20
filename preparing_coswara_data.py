import pandas as pd
import pathlib
from pathlib import Path
import os
import shutil
import json
import glob


def gen_dataframe(file_path):
    '''
    create dataframe and append values to it
    '''

    #creating empty lists 
    ages = []
    cov_stat = []
    genders = []
    wav_paths = []
    id = []
    date = []
    
    #empty dataframe
    df = pd.DataFrame(columns=["ID","Age","Covid_status","Gender", "wav_path"])

    for a in glob.glob(file_path +"/*"):
        d = a.split('\\')[-1]
        for fol in glob.glob(a + "/*"): 
            if os.path.exists(f"{fol}/metadata.json") == False:
                break
            full_wav_path = f"{fol}/cough-heavy.wav"
            wav_paths.append(full_wav_path)
            id.append(fol.split('\\')[-1])
            with open(f"{fol}/metadata.json") as json_file:  
                data = json.load(json_file)
                ages.append(data['a'])
                cov_stat.append(data['covid_status'])
                genders.append(data['g'])
                date.append(d)
    df['ID'] = id
    df['Age'] = ages
    df['Covid_status'] = cov_stat
    df['Gender'] = genders
    df['wav_path'] = wav_paths
    df['date'] = date
     
    maps = {
            "healthy":"negative",
            "no_resp_illness_exposed":"negative",
            "positive_moderate":"positive",
            "positive_asymp":"positive",
            "positive_mild":"positive",
            "recovered_full":"negative",
            "resp_illness_not_identified":"Unknown"
           }

    df['Covid_status'] = df['Covid_status'].map(maps)
    df['nums'] = list(range(df.shape[0]))
    df['new_name'] = file_path + "/" + df['date'] + "/" + df['ID'] + "/" + df['nums'].astype(str)+ "_" + df['Covid_status']+ "_" + df['Gender']+ "_" + df['Age'].astype(str) + ".wav"
    df.drop(['nums'],axis=1,inplace=True)
    
    return df


def rename_move_files(df, source_path, dest_path):
    # renaming the files 
    old_names = list(df['wav_path'])
    new_names = list(df['new_name'])
    for idx, e in enumerate(old_names):
        os.rename(e, new_names[idx])
    results = ['positive', 'negative', 'Unknown']
    for res in results:
        pathlib.Path(source_path + f"/{res}").mkdir(parents=True, exist_ok=True)
        for files in Path(source_path + "/").rglob("*.wav"):
            # print(files)
            # print(res in files.parts[-1])
            if res in files.parts[-1]:
                dest = f"{dest_path}/{res}/{files.parts[-1]}"
                shutil.move(files, dest)


if __name__=='__main__':
    df = gen_dataframe('./data/Extracted_data')
    rename_move_files(df, './data/Extracted_data', './data/cleaned_data')
    pos = df[df['Covid_status'] == "positive"]
    neg = df[df['Covid_status'] == "negative"]
    unk = df[df['Covid_status'] == "Unknown"]
    print(f'positive:{pos.shape[0]}\n'
          f'Negative:{neg.shape[0]}\n'
          f'Unknown:{unk.shape[0]}\n')