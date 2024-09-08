package com.example.thesisocr

import android.graphics.Bitmap
import org.opencv.android.Utils
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc

// Tasked with detecting document boundaries in an image and cropping the image to the detected boundaries.
// Use OpenCV for image processing.

class DocumentScan {
    data class ScanImage(
        var image: Bitmap
    )
    fun scanDocument(inputBitmap: Bitmap): ScanImage {
        val inputMat = convertToMat(inputBitmap)
        val matPreprocessed = imagePreprocessing(inputMat)
        val edgeDetectedMat = edgeDetection(matPreprocessed)
        val contours = findContours(edgeDetectedMat)
        val outputMat = drawContours(inputMat, contours)
        val outputBitmap = matToBitmap(outputMat)
        return ScanImage(outputBitmap)
    }
    private fun convertToMat(inputBitmap: Bitmap): Mat {
        val inputMat = Mat()
        Utils.bitmapToMat(inputBitmap, inputMat)
        return inputMat
    }
    private fun imagePreprocessing(inputMat: Mat): Mat {
        // Convert to grayscale.
        val grayMat = Mat()
        Imgproc.cvtColor(inputMat, grayMat, Imgproc.COLOR_BGR2GRAY)
        // Apply Gaussian Blur.
        val blurredMat = Mat()
        Imgproc.GaussianBlur(grayMat, blurredMat, Size(5.0, 5.0), 0.0)
        // Apply Canny Edge Detection.
        val edgesMat = Mat()
        Imgproc.Canny(blurredMat, edgesMat, 75.0, 200.0)
        return edgesMat
    }
    private fun edgeDetection(inputMat: Mat): Mat {
        // Apply Canny Edge Detection.
        val edgesMat = Mat()
        Imgproc.Canny(inputMat, edgesMat, 75.0, 200.0)
        return edgesMat
    }
    private fun findContours(inputMat: Mat): List<MatOfPoint> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(inputMat, contours, hierarchy, Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE)
        return contours
    }
    // Debug functions
    private fun matToBitmap(inputMat: Mat): Bitmap {
        val outputBitmap = Bitmap.createBitmap(inputMat.width(), inputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(inputMat, outputBitmap)
        return outputBitmap
    }
    private fun drawContours(inputMat: Mat, contours: List<MatOfPoint>): Mat {
        Imgproc.drawContours(inputMat, contours, -1, Scalar(0.0, 255.0, 0.0), 3)
        return inputMat
    }
}