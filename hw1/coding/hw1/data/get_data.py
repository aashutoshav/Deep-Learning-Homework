import os
import gdown
import tarfile


def download_and_extract(url, filename):
    print(f"Downloading {url}...")
    gdown.download(url, filename, quiet=False, fuzzy=True)

    print(f"Extracting {filename}...")
    with tarfile.open(filename, "r:gz") as tar:
        tar.extractall(path=".")  # extract into current directory

    print(f"Removing {filename}...")
    os.remove(filename)


test_url = "https://drive.google.com/file/d/1gX1Y_6EeL-z7bDgf2qU8wb1zKlgUMKQn/view?usp=share_link"
train_url = "https://drive.google.com/file/d/1t_6j3bfTCcbeVRm0CUhc-co3IoEKQ04d/view?usp=share_link"

# Download and extract the test set
download_and_extract(test_url, "mnist_test.tar.gz")

# Download and extract the train set
download_and_extract(train_url, "mnist_train.tar.gz")

print("Done! Files are in the 'data' folder.")
