package com.example.thesisocr

import android.graphics.Bitmap
import android.os.Environment
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.opencv.utils.Converters
import java.util.Arrays
import kotlin.math.pow
import kotlin.math.sqrt

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
    fun imagePreProcess(bitmap: Bitmap): Bitmap {
        val inputMat = Mat()
        Utils.bitmapToMat(bitmap, inputMat)
        Log.d("Input Mat:", "$inputMat")
        val edgeImage = cannyEdge(inputMat)
        val houghLines = houghTransform(edgeImage, inputMat)
        val intersections = getIntersection(houghLines, inputMat)
        val bestQuad = computeQuadrilateralScore(intersections)
        val outputMat = perspectiveTransform(inputMat, bestQuad)
        val outputBitmap = Bitmap.createBitmap(outputMat.width(), outputMat.height(), Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(outputMat, outputBitmap)
        Log.d("Output Bitmap:", "$outputBitmap")
        return outputBitmap
    }
    // Canny Edge Detection
    fun cannyEdge(inputImage: Mat): Mat {
        // Function Variables
        val grayImage = Mat()
        val threshImage = Mat()
        val morphImage = Mat()
        val edges = Mat()
        // Canny parameters
        val thresholdOne = 255.0
        val thresholdTwo = 255.0/3.0
        try {
            // Convert to grayscale
            Imgproc.cvtColor(inputImage, grayImage, Imgproc.COLOR_BGR2GRAY)
            // Threshold
            Imgproc.GaussianBlur(grayImage,grayImage,Size(5.0,5.0),0.0)
            Imgproc.threshold(grayImage,threshImage, 0.0, 255.0,Imgproc.THRESH_OTSU)
            // Run OpenCV Canny
            Imgproc.Canny(threshImage, edges, thresholdOne, thresholdTwo)
            saveMatAsJpg(edges, Environment.getExternalStorageDirectory().path+"/Pictures/edgeImage.jpg")
            Log.e("Canny Edge:", "Success!")
            Log.d("Canny Edge Output: ", "$edges")
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("Canny Edge:", "Failure!")
        }
        return edges
    }
    // Standard Hough Line Transform
    fun houghTransform(inputMat: Mat, inputImage: Mat): Mat {
        Log.d("Hough Transform Input Details:", "$inputMat")
        val houghLines = Mat() // Output variable
        val rho = 1.0
        val theta = Math.PI/180
        try {
            // DO NOT SET THRESHOLD TO BELOW 128
            Imgproc.HoughLinesP(inputMat, houghLines, rho, theta, 200, 5.0, 1.0)
            Log.d("Hough Lines Rows:", "${houghLines.rows()}")
            for (x in 0..<houghLines.rows()) {
                val l = houghLines.get(x,0)
                Imgproc.line(
                    inputImage, Point(l[0], l[1]), Point(
                        l[2],
                        l[3]
                    ), Scalar(0.0, 0.0, 255.0), 3, Imgproc.LINE_AA, 0
                )
            }
            Imgcodecs.imwrite(Environment.getExternalStorageDirectory().path+"/Pictures/houghImage.jpg", inputImage)
            Log.e("Hough Transform:", "Success!")
        } catch (e: Exception) {
            e.printStackTrace()
            Log.e("Hough Transform:", "Failure!")
        }
        return houghLines
    }
    fun getIntersection(houghLines: Mat, inputMat: Mat): MutableList<Point> {
        Log.d("Hough Lines:", "$houghLines")
        val intersections = mutableListOf<Point>()
        for (i in 0 ..<houghLines.rows()) {
            Log.d("Hough Line at $i:", "${houghLines.get(i, 0)}")
            val l1 = houghLines.get(i, 0)
            val p1 = Point(l1[0], l1[1])
            val p2 = Point(l1[2], l1[3])
            for (j in i + 1 until houghLines.rows()) {
                val l2 = houghLines.get(j, 0)
                val p3 = Point(l2[0], l2[1])
                val p4 = Point(l2[2], l2[3])
                val intersection = calculateIntersection(p1, p2, p3, p4)
                if (intersection != null) {
                    intersections.add(intersection)
                    Imgproc.circle(inputMat, intersection, 25, Scalar(0.0, 255.0, 0.0), 5)
                    Log.d("Intersection value at $i:", "${intersections[i]}")
                } else {
                    Log.d("Intersection value at $i:", "No intersection found.")
                }
            }
        }
        // Render intersection points for debugging
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory().path+"/Pictures/intersectionImage.jpg", inputMat)
        Log.d("Intersections:","Success!")
        Log.d("Intersections:", "$intersections")
        return intersections
    }
    fun computeQuadrilateralScore(intersections: MutableList<Point>): Mat{
        Log.d("Quadrilateral Score Input:", "$intersections")
        var maxScore = 0.0
        val bestQuad = Mat(4,1, CvType.CV_32FC2)
         // Compute scores
        for (i in 0 until intersections.size) {
            for (j in i + 1 until intersections.size) {
                for (k in j + 1 until intersections.size) {
                    for (l in k + 1 until intersections.size) {
                        val score = computeScore(intersections[i], intersections[j], intersections[k], intersections[l])
                        if (score > maxScore) {
                            maxScore = score
                            bestQuad.put(0,0,intersections[i].x,intersections[i].y)
                            bestQuad.put(1,0,intersections[j].x,intersections[j].y)
                            bestQuad.put(2,0,intersections[k].x,intersections[k].y)
                            bestQuad.put(3,0,intersections[l].x,intersections[l].y)
                        }
                        Log.d("Quadrilateral Score:", "$score")
                    }
                }
            }
        }
        Log.d("Best Quadrilateral Score:", "$maxScore")
        Log.d("Best Quadrilateral:", "$bestQuad")
        return bestQuad
    }
    fun perspectiveTransform(inputMat: Mat, bestQuad: Mat): Mat {
        Log.d("Perspective Transform Input Details:", "$inputMat")
        val outputMat = Mat()
        val inputQuad = Mat(4,1, CvType.CV_32FC2)
        val outputQuad = mutableListOf<Point>()
        Log.d("Perspective Transform Best Quad:", "$bestQuad")
        // Convert bestQuad to inputQuad
        for (i in 0 until bestQuad.rows()) {
            val point = Point(bestQuad.get(i,0)[0],bestQuad.get(i,0)[1])
            inputQuad.put(i,0,point.x,point.y)
        }
        Log.d("Perspective Transform Input Quad:", "$inputQuad")
        // Define the destination image
        val outputSize = Size(inputMat.size().width, inputMat.size().height)
        outputQuad.add(Point(0.0, 0.0))
        outputQuad.add(Point(outputSize.width - 1, 0.0))
        outputQuad.add(Point(outputSize.width - 1, outputSize.height - 1))
        outputQuad.add(Point(0.0, outputSize.height - 1))
        // Get the Perspective Transform Matrix
        val perspectiveTransform: Mat = Imgproc.getPerspectiveTransform(bestQuad, Converters.vector_Point2f_to_Mat(outputQuad))
        Log.d("Point2f_to_Mat:", "${Converters.vector_Point2f_to_Mat(outputQuad)}")
        Log.d("Perspective Transform Matrix:", "$perspectiveTransform")
        // Apply the Perspective Transform just found to the input image
        if (inputMat.empty()) {
            Log.e("Warp Perspective","Input image is empty")
        } else {
            Imgproc.warpPerspective(inputMat, outputMat, perspectiveTransform, outputSize)
            Log.d("Perspective Transform Output:", "$outputMat")
        }
        return outputMat
    }
    private fun calculateIntersection(p1: Point, p2: Point, q1: Point, q2: Point): Point? {
        val det = (p2.x - p1.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q2.x - q1.x)
        if (det == 0.0) {
            return null
        }
        val x = ((p2.x * p1.y - p2.y * p1.x) * (q2.x - q1.x) - (p2.x - p1.x) * (q2.x * q1.y - q2.y * q1.x)) / det
        val y = ((p2.x * p1.y - p2.y * p1.x) * (q2.y - q1.y) - (p2.y - p1.y) * (q2.x * q1.y - q2.y * q1.x)) / det
        return Point(x, y)
    }
    private fun computeScore(point: Point, point1: Point, point2: Point, point3: Point): Double {
        val side1 = sqrt((point.x - point1.x).pow(2) + (point.y - point1.y).pow(2))
        val side2 = sqrt((point1.x - point2.x).pow(2) + (point1.y - point2.y).pow(2))
        val side3 = sqrt((point2.x - point3.x).pow(2) + (point2.y - point3.y).pow(2))
        val side4 = sqrt((point3.x - point.x).pow(2) + (point3.y - point.y).pow(2))
        val semiPerimeter = (side1 + side2 + side3 + side4) / 2
        return sqrt(
            semiPerimeter * (semiPerimeter - side1) * (semiPerimeter - side2) * (semiPerimeter - side3) * (semiPerimeter - side4)
        )
    }
    fun saveMatAsJpg(mat: Mat, fileName: String) {
        val outputImage = Mat()
        Imgproc.cvtColor(mat, outputImage, Imgproc.COLOR_GRAY2BGR)
        Imgcodecs.imwrite(fileName, outputImage)
        Log.d("Image Saved:", fileName)
    }
}