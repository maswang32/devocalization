import argparse
import os
import librosa
import numpy as np


"""
File for Processing the Vocal Imitation Dataset
"""

DATA_DIR = "/viscam/projects/audio_nerf/transfer/audio_nerf/data"
RAW_DIR = os.path.join(DATA_DIR, "VocalImitationSet")
REF_DIR = os.path.join(RAW_DIR, "original_recordings/reference")
IM_DIR = os.path.join(RAW_DIR, "vocal_imitations/included")
RATINGS_PATH = os.path.join(RAW_DIR, "vocal_imitations_assessment.txt")
CATEGORIES_PATH = os.path.join(RAW_DIR, "vocal_imitations.txt")


def filter_by_threshold(categories_dict, threshold = 50):
    """Returns paths for imitation and reference recordings filtered by rating threshold"""
    imitation_paths = []
    reference_paths = []
    ratings = []
    category_ids = []

    with open(RATINGS_PATH, 'r') as file:
        next(file) # Skip first line

        for line in file:
            parts = line.strip().split('\t')
            try:
                rating = float(parts[-1])
            except:
                pass

            if rating > threshold and parts[2] not in imitation_paths: #Filter out duplicates

                print("\n\nRating\t:" + parts[-1])
                print("Imitation Filename:\t" + parts[2])
                print("Reference Filename:\t" + parts[3])
                print("Category ID:\t" + str(categories_dict[parts[3]]))

                ratings.append(rating)
                imitation_paths.append(parts[2])
                reference_paths.append(parts[3])
                category_ids.append(categories_dict[parts[3]])

    print(len(imitation_paths))
    assert len(imitation_paths) == len(reference_paths) == len(ratings)
    return imitation_paths, reference_paths, category_ids, ratings



def process_data(imitation_paths,
                reference_paths,
                category_ids,
                fs=44100, 
                trim=True,
                min_length = 3,
                full_length = 6):
    """Reads and Processes Audio Files"""
    num_samples = int(full_length*fs)
    imitation_recordings = np.zeros((len(imitation_paths), num_samples))
    reference_recordings = np.zeros((len(reference_paths), num_samples))
    filtered_category_ids = []
    filtered_reference_paths = []

    valid_count = 0

    for i, path in enumerate(imitation_paths):

        imitation_path = os.path.join(IM_DIR, imitation_paths[i])
        reference_path = os.path.join(REF_DIR, reference_paths[i])

        if os.path.isfile(imitation_path) and os.path.isfile(reference_path):
            im, _ = librosa.load(imitation_path, sr=fs) # Resamples, converts to float, and converts to mono
            ref, _ = librosa.load(reference_path, sr=fs)

            if trim:
                im, _ = librosa.effects.trim(im)
                ref, _ = librosa.effects.trim(ref)


            length_s_im = im.shape[0]/fs
            length_s_ref = ref.shape[0]/fs

            if length_s_im < min_length or length_s_ref < min_length:
                print("Recording Too short")
                continue
            else:
                imitation_recordings[valid_count, :min(num_samples, im.shape[-1])] = im[...,:num_samples]
                reference_recordings[valid_count, :min(num_samples, ref.shape[-1])] = ref[...,:num_samples]
                filtered_category_ids.append(category_ids[i])
                filtered_reference_paths.append(reference_path)
                valid_count += 1
        else:
            print('Recording not found')

        if i%100 == 0:
            print(i)


    assert len(filtered_category_ids) == valid_count
    print("Number of Examples:\t" + str(valid_count))
    imitation_recordings = imitation_recordings[:valid_count]
    reference_recordings = reference_recordings[:valid_count]

    return imitation_recordings, reference_recordings, filtered_category_ids, filtered_reference_paths




if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("save_name")
    parser.add_argument("--fs", type=int, default=44100)
    parser.add_argument("--full_length", type=float, default=6)
    args = parser.parse_args()

    
    # Making a dictionary mapping paths to category ids
    categories_dict = {}
    with open(CATEGORIES_PATH, 'r') as file:
        next(file) # Skip first line

        for line in file:
            parts = line.strip().split('\t')
            reference_path = parts[10]
            category_id = int(parts[3])
            categories_dict[reference_path] = category_id

    # Getting Paths
    imitation_paths, reference_paths, category_ids, ratings = filter_by_threshold(categories_dict)

    # Getting Recordings
    imitation_recordings, reference_recordings, filtered_category_ids, filtered_reference_paths = process_data(
        imitation_paths, reference_paths, category_ids, full_length=args.full_length, fs=args.fs)
    
    # Saving
    save_dir = os.path.join(DATA_DIR, args.save_name)
    np.save(os.path.join(save_dir, "imitations.npy"), imitation_recordings)
    np.save(os.path.join(save_dir, "reference.npy"), reference_recordings)
    np.save(os.path.join(save_dir, "categories.npy"), filtered_category_ids)
    np.save(os.path.join(save_dir, "reference_paths.npy"), filtered_reference_paths)
    print(f"Number of Distinct Reference Paths:\t{len(set(reference_paths))}")




    




