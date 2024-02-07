package com.example.thesisocr

import android.Manifest
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import org.opencv.android.OpenCVLoader
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc

class MainActivity : AppCompatActivity() {
    // TODO: Integrate PreProcessing.
    // TODO: Move camera call and file picker functions to separate class.
    private var imageView: ImageView? = null
    private val preProcessing = PreProcessing()
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            imagePreProcess(bitmap)
            Log.d("Photo Picker", "Photo selected: $uri")
        } else {
            Log.d("Photo Picker", "No photo selected.")
        }
    }
    override fun onCreate(savedInstanceState: Bundle?) {
        // Load OpenCV
        OpenCVLoader.initLocal()
        if(OpenCVLoader.initLocal()){
            Log.e("MyTag","OpenCV Loaded.")
        } else {
            Log.e("MyTag","OpenCV Not Loaded.")
        }
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        val btnCapture = findViewById<Button>(R.id.btnCapture)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        imageView = findViewById(R.id.imageView)
        btnCapture.setOnClickListener {
            if (checkCameraPermission()) {
                openCamera()
            } else {
                requestCameraPermission()
            }
        }
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
    }

    private fun checkCameraPermission(): Boolean {
        return ContextCompat.checkSelfPermission(
            this,
            Manifest.permission.CAMERA
        ) == PackageManager.PERMISSION_GRANTED
    }

    private fun requestCameraPermission() {
        ActivityCompat.requestPermissions(
            this, arrayOf(Manifest.permission.CAMERA),
            CAMERA_PERMISSION_REQUEST
        )
    }
    private fun openCamera() {
        val captureIntent = Intent(MediaStore.ACTION_IMAGE_CAPTURE)
        startActivityForResult(
            captureIntent,
            IMAGE_CAPTURE_REQUEST
        ) // TODO: Replace deprecated function.
    }
    override fun onRequestPermissionsResult(
        requestCode: Int,
        permissions: Array<String>,
        grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.isNotEmpty() && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                openCamera()
            }
        }
    }

    @Deprecated("Deprecated in Java")
    override fun onActivityResult(requestCode: Int, resultCode: Int, data: Intent?) {
        super.onActivityResult(requestCode, resultCode, data)
        if (resultCode == RESULT_OK) {
            if (requestCode == IMAGE_CAPTURE_REQUEST) {
                // Handle the captured image, e.g., save or display it
                val imageBitmap = data!!.extras!!["data"] as Bitmap?
                displayImage(imageBitmap)
            } else if (requestCode == IMAGE_PICK_REQUEST) {
                // Handle the selected image from the file explorer
                val selectedImageUri = data!!.data
                displayImageFromUri(selectedImageUri)
                val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, selectedImageUri)
                imagePreProcess(bitmap)
            }
        }
    }

    private fun imagePreProcess(bitmap: Bitmap){
        val bitmapAsMat = Mat()
        Utils.bitmapToMat(bitmap, bitmapAsMat)
        Imgproc.cvtColor(bitmapAsMat, bitmapAsMat, Imgproc.COLOR_BGRA2BGR)
        Log.d("Image Bitmap Output","$bitmapAsMat")
        Log.d("Image Bitmap", "Image Bitmap to Mat Conversion Successful. ${bitmapAsMat.size()}")
        Log.d("Image Pre-Processing", "Image Pre-Processing Started.")
        val edgeImage = preProcessing.cannyEdge(bitmapAsMat)
        val houghImage = preProcessing.houghTransform(edgeImage)
        val intersections = preProcessing.getIntersection(houghImage)
        val bestQuad = preProcessing.computeQuadrilateralScore(intersections)
        val outputMat = preProcessing.perspectiveTransform(bitmapAsMat, bestQuad)
        Imgcodecs.imwrite(Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg", outputMat)
        Log.d("Image Pre-Processing", "Image Pre-Processing Completed.")
        Log.d("Output Image", "Output Image Saved to ${Environment.getExternalStorageDirectory().toString() + "/Pictures/output.jpg"}")
    }

    private fun displayImage(bitmap: Bitmap?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageBitmap(bitmap)
    }

    private fun displayImageFromUri(imageUri: Uri?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageURI(imageUri)
    }

    companion object {
        private const val CAMERA_PERMISSION_REQUEST = 100
        private const val IMAGE_CAPTURE_REQUEST = 101
        private const val IMAGE_PICK_REQUEST = 102
    }
}