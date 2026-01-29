"""Sort the data set into training, validation, testing"""
import hashlib
import os
import re
from pathlib import Path

MAX_NUM_WAVS_PER_CLASS = 2**27 - 1  # ~134M
def which_set(filename, validation_percentage, testing_percentage):
    """Determines which data partition the file should belong to.
    
    We want to keep files in the same training, validation, or testing sets even
    if new ones are added over time. This makes it less likely that testing
    samples will accidentally be reused in training when long runs are restarted
    for example. To keep this stability, a hash of the filename is taken and used
    to determine which set it should belong to. This determination only depends on
    the name and the set proportions, so it won't change as other files are added.
    
    It's also useful to associate particular files as related (for example words
    spoken by the same person), so anything after '_nohash_' in a filename is
    ignored for set determination. This ensures that 'bobby_nohash_0.wav' and
    'bobby_nohash_1.wav' are always in the same set, for example.
    
    Args:
      filename: File path of the data sample.
      validation_percentage: How much of the data set to use for validation.
      testing_percentage: How much of the data set to use for testing.
    
    Returns:
      String, one of 'training', 'validation', or 'testing'.
    """
    base_name = os.path.basename(filename)
    # We want to ignore anything after '_nohash_' in the file name when
    # deciding which set to put a wav in, so the data set creator has a way of
    # grouping wavs that are close variations of each other.
    hash_name = re.sub(r'_nohash_.*$', '', base_name)
    # This looks a bit magical, but we need to decide whether this file should
    # go into the training, testing, or validation sets, and we want to keep
    # existing files in the same set even if more files are subsequently
    # added.
    # To do that, we need a stable way of deciding based on just the file name
    # itself, so we do a hash of that and then use that to generate a
    # probability value that we use to assign it.
    hash_name_hashed = hashlib.sha1(hash_name.encode()).hexdigest()
    percentage_hash = ((int(hash_name_hashed, 16) %
                        (MAX_NUM_WAVS_PER_CLASS + 1)) *
                       (100.0 / MAX_NUM_WAVS_PER_CLASS))
    if percentage_hash < validation_percentage:
      result = 'validation'
    elif percentage_hash < (testing_percentage + validation_percentage):
      result = 'testing'
    else:
      result = 'training'
    return result

def main():
    dirs = [p for p in Path("./speech-data").iterdir() if p.is_dir()]
    with (open("train_pos_set.txt", "w") as tra_pos_f,
          open("train_neg_set.txt", "w") as tra_neg_f,
          open("val_pos_set.txt", "w") as val_pos_f,
          open("val_neg_set.txt", "w") as val_neg_f,
          open("test_pos_set.txt", "w") as tes_pos_f,
          open("test_neg_set.txt", "w") as tes_neg_f):
        for dir_ in dirs:
            for p in dir_.iterdir():
                if not p.name.endswith(".wav"): continue
                set_choice = which_set(p.name, 20, 0)
                if dir_.name.startswith("stop"):
                    if set_choice == "validation":
                        val_pos_f.write(f"{dir_.name}/{p.name}\n")
                    elif set_choice == "testing":
                        tes_pos_f.write(f"{dir_.name}/{p.name}\n")
                    if set_choice == "training":
                        tra_pos_f.write(f"{dir_.name}/{p.name}\n")
                else:
                    if set_choice == "validation":
                        val_neg_f.write(f"{dir_.name}/{p.name}\n")
                    elif set_choice == "testing":
                        tes_neg_f.write(f"{dir_.name}/{p.name}\n")
                    if set_choice == "training":
                        tra_neg_f.write(f"{dir_.name}/{p.name}\n")

if __name__ == "__main__":
   main()
