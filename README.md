# Sybrin Digital Onboarding & Document Intelligence API

## Project Overview
This repository contains a specialized Machine Learning API designed for **Digital Onboarding** and **KYC (Know Your Customer)** compliance. Inspired by Sybrin's focus on financial services in Africa, this tool automates the extraction of data from identification documents.

## Key Features
* **Document Segmentation:** Uses OpenCV to detect the physical boundaries of an ID card in a photo.
* **Perspective Transformation:** Automatically corrects the "tilt" of a photo, warping it into a flat, scannable rectangle.
* **OCR Engine:** Leverages Deep Learning (EasyOCR/PyTorch) to extract text and numbers from the processed document.
* **Production-Ready API:** Built with FastAPI, demonstrating an understanding of how ML models are deployed in real-world software environments.

## Technical Stack
* **Language:** Python 3.10
* **Computer Vision:** OpenCV
* **Deep Learning:** PyTorch & EasyOCR
* **Backend:** FastAPI & Uvicorn

## How to Run
1. Clone the repository.
2. Install dependencies: `pip install fastapi uvicorn opencv-python easyocr torch`.
3. Run the server: `python main.py`.
4. Access the API documentation at `http://127.0.0.1:8000/docs`.