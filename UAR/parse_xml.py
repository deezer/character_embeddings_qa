from collections import defaultdict
from bs4 import BeautifulSoup
import re 
import os 
import argparse 
import json 
from tqdm import tqdm 

def xml_to_json(file, segment_unit=None):
    """Parse gutenberg XML file to a json-style python dictionary.
    args:
    - file: path to xml file"""
    assert segment_unit in [None, "scene", "acts"]
    
    with open(file, "r") as f  :
        text = f.read()
    soup = BeautifulSoup(text, 'xml')
    
    play_name = soup.find("title")
    if play_name is not None : 
        play_name = play_name.string
    else : 
        play_name = ""
    # play_name = soup.find("title").string if soup.find("title").string is not None else ""
    nationality = soup.find("nationality")
    if nationality is not None :
        if nationality.string not in ["English", "American", "Irish", "Scottish", "Welsh", "New Zealander", "French"]  : 
            return 0, None, None
    else : 
        return 0, None, None

    author_forename = soup.find("forename")
    if author_forename is not None : 
        author_forename = author_forename.string
    else : 
        author_forename = ""
    # author_forename = soup.find("forename").string if soup.find("forename").string is not None else ""
    author_surname = soup.find("surname")
    if author_surname is not None : 
        author_surname = author_surname.string
    else : 
        author_surname = ""
    # author_surname =  soup.find("surname").string if soup.find("surname").string is not None else ""
    try : 
        author = author_forename + " " + author_surname
    except : 
        author = ""
        
    publication_date = soup.find("date").string
    meta_data = {"author": author, "publication_date":publication_date, "filename": os.path.split(file)[-1]}
    types = set([i.attrs["type"] for i in soup.find_all(type=True)])
    # if "act" in types : 
    #     # Default to act unit
    #     acts = soup.find_all(type="act")
    #     meta_data["segment_unit"] = "Act"
    # else : 
    #     if "scene" in types : 
    #         # If no act, then back to scene
    #         acts = soup.find_all(type="scene")
    #         meta_data["segment_unit"] = "Scene"
    #     else : 
    #         # no segementation unit available, default to full play
    #         return 0, None, None
    if not segment_unit :
        acts = [soup]
        meta_data["segment_unit"] = "full"
        
    elif segment_unit == "scene" : 
        if "scene" in types : 
            # Default to scene unit
            acts = soup.find_all(type="scene")
            meta_data["segment_unit"] = "Scene"
        else : 
            if "acts" in types : 
                # If then act, then back to scene
                acts = soup.find_all(type="acts")
                meta_data["segment_unit"] = "Acts"
            else : 
                # no segementation unit available
                return 0, None, None
    elif segment_unit == "act" : 
        if "act" in types : 
            # Default to act unit
            acts = soup.find_all(type="act")
            meta_data["segment_unit"] = "Act"
        else : 
            if "scene" in types : 
                # If no act, then back to scene
                acts = soup.find_all(type="scene")
                meta_data["segment_unit"] = "Scene"
            else : 
                # no segementation unit available
                return 0, None, None
            
    chars = set([i.string.upper() for i in soup.find_all("speaker")])
    char_mapper = {char:i for char,i in zip(chars, range(len(chars)))}
    json_data = {"char_mapper":char_mapper, "metadata":meta_data}
    data = []
    for idx, act in enumerate(acts) :
        act_data = defaultdict(list)
        all_lines = act.find_all("sp")
        for lines_by_speaker in all_lines: 
            speaker = lines_by_speaker.find("speaker").string.upper()
            speaker_id = char_mapper[speaker]
            for line in lines_by_speaker.find_all(["p", "l"]) : 
                try : 
                    act_data[speaker_id].append(line.string.strip())
                except : 
                    pass
        data.append(act_data)
    json_data["acts"] = data
    
    return 1, play_name, json_data

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_path", help="path to folder containing all xml files")
    parser.add_argument("--dest_path", help="path where data will be saved")
    parser.add_argument("--segment-unit", help="semgent unit to chose. Default to None", default = None)

    args = parser.parse_args()
    
    full_data = {}
    valid = 0 
    total = 0 
    for dir in tqdm(os.listdir(args.source_path)) : 
        file = os.path.join(args.source_path, dir)
        is_valid, play, data = xml_to_json(file, segment_unit = args.segment_unit)
        
        if is_valid == 1 : 
            full_data[play] = data
            valid += 1 
        total += 1

    print(f"Total number of plays {total}")
    print(f"Parsed number of plays {valid}")
    
    with open(args.dest_path, "w") as f : 
        json.dump(full_data, f)
        