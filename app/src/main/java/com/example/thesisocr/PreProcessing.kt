package com.example.thesisocr

import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import kotlin.math.cos
import kotlin.math.sin

class PreProcessing {
    // DONE: Implement Canny Edge Detection
    // DONE: Implement Hough Transform
    // TODO: Implement class and function calls on MainActivity.kt

    /*
     Call the functions in order of: cannyEdge -> houghTransform().
     Use any available functions to convert resulting Mat() to Bitmap
     for use within Android UI framework.
     */

    companion object {
        init {
            System.loadLibrary("opencv_java4")
        }
    }
    // Canny Edge Detection
    fun cannyEdge(inputBitmap: Bitmap): Mat {
        // Function Variables
        val inputImage = Mat()
        val grayImage = Mat()
        val threshImage = Mat()
        val morphImage = Mat()
        val edges = Mat()
        // Canny parameters
        val thresholdOne = 255.0
        val thresholdTwo = 255.0/3.0
        try {
            // Convert Bitmap input into Mat
            Utils.bitmapToMat(inputBitmap, inputImage)
            // Convert to grayscale
            Imgproc.cvtColor(inputImage, grayImage, Imgproc.COLOR_BGR2GRAY)
            // Threshold
            Imgproc.GaussianBlur(grayImage,grayImage,Size(5.0,5.0),0.0)
            Imgproc.threshold(grayImage,threshImage, 0.0, 255.0,Imgproc.THRESH_OTSU)
            // Clean through Morphology
            // Imgproc.morphologyEx(threshImage, morphImage, Imgproc.MORPH_OPEN, Mat.ones(Size(15.0,1.0), CvType.CV_8U))
            // Imgproc.morphologyEx(morphImage, morphImage, Imgproc.MORPH_CLOSE, Mat.ones(Size(17.0,3.0), CvType.CV_8U))
            // Run OpenCV Canny
            Imgproc.Canny(threshImage, edges, thresholdOne, thresholdTwo)
            saveMatAsJpg(edges, Environment.getExternalStorageDirectory().path+"/Pictures/edgeImage.jpg")
            // Release Mat objects
            inputImage.release()
            grayImage.release()
            threshImage.release()
            morphImage.release()
            Log.e("Canny Edge:", "Success!")
            Log.d("Canny Edge Output: ", "$edges")
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("Canny Edge:", "Failure!")
        }
        return edges
    }
    // Standard Hough Line Transform
    fun houghTransform(inputMat: Mat): Mat {
        Log.d("Hough Transform Input Details:", "$inputMat")
        val houghLines = Mat() // Output variable
        val rho = 1.0
        val theta = Math.PI/180
        try {
            Imgproc.HoughLinesP(inputMat, houghLines, rho, theta, 165, 5.0, 5.0)
            Log.d("Hough Transform:", "$houghLines")
            for (x in 0..<houghLines.rows()) {
                val l = houghLines.get(x,0)
                Imgproc.line(
                    inputMat, Point(l[0], l[1]), Point(
                        l[2],
                        l[3]
                    ), Scalar(0.0, 0.0, 255.0), 3, Imgproc.LINE_AA, 0
                )
            }
            Log.d("Hough Transform:", "$inputMat")
            saveMatAsJpg(inputMat,Environment.getExternalStorageDirectory().path+"/Pictures/houghImage.jpg")
            Log.e("Hough Transform:", "Success!")
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("Hough Transform:", "Failure!")
        }
        return houghLines
    }
    // TODO: Implement algorithm that takes advantage of the above two to perform image transformation.
    fun imageTransformation(inputMat: Mat, line1: Pair<Double,Double>, line2: Pair<Double, Double>): Mat? {
        var lines = listOf(listOf(0.0, 0.0))
        val (r1, t1) = line1
        val (r2, t2) = line2
        val a = Mat(2, 2, CvType.CV_64F)
        a.put(0, 0, Math.cos(t1))
        a.put(0, 1, Math.sin(t1))
        a.put(1, 0, Math.cos(t2))
        a.put(1, 1, Math.sin(t2))
        val b = Mat(2, 1, CvType.CV_64F)
        b.put(0, 0, r1)
        b.put(1, 0, r2)
        val x = Mat(2, 1, CvType.CV_64F)
        Core.solve(a, b, x)
        val x0 = x.get(0, 0)[0]
        val y0 = x.get(1, 0)[0]
        if (Math.abs(t1 - t2) > 1.3) {
            lines = listOf(listOf(x0, y0))
            Log.d("Image Transformation", "Lines Intersection Done")
            Log.d("Image Transformation:", "$lines")
        } else {
            Log.e("Image Transformation","Error at Lines Intersection")
        }
        // TODO: Implement Points Intersection
        return null
    }
    private fun saveMatAsJpg(mat: Mat, fileName: String) {
        val outputImage = Mat()
        Imgproc.cvtColor(mat, outputImage, Imgproc.COLOR_GRAY2BGR)
        Imgcodecs.imwrite(fileName, outputImage)
        Log.d("Image Saved:", fileName)
        outputImage.release()
    }
}