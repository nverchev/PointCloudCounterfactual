"""Functions for downloading data."""

import logging
import pathlib
import zipfile

import requests


def download_extract_zip(target_folder: pathlib.Path, url: str) -> None:
    """Downloads and extracts a zip file from a URL to a target folder."""
    logging.info(f'Checking if folder exists: {target_folder}')

    if not target_folder.exists():
        logging.info(f'Folder does not exist. Starting download from: {url}')
        r = requests.get(url)
        logging.info(f'Download complete. Size: {len(r.content)} bytes')
        zip_path = target_folder.with_suffix('.zip')
        logging.info(f'Saving zip file to: {zip_path}')

        with zip_path.open('wb') as zip_file:
            zip_file.write(r.content)

        logging.info(f'Zip file saved successfully. Extracting to: {target_folder.parent}')
        with zipfile.ZipFile(zip_path) as zip_ref:
            zip_ref.extractall(target_folder.parent)

        logging.info('Extraction complete')
    else:
        logging.info(f'Folder already exists at {target_folder}. Skipping download.')

    return
