package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.ContentValues
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.view.View
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.CameraSelector
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import com.example.thesisocr.databinding.ActivityCameraBinding
import org.opencv.android.OpenCVLoader
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class MainActivity : AppCompatActivity() {
    // Model Vocabulary from en_dict.txt raw resource file.
    private lateinit var modelVocab: List<String>
    // ONNX Variables
    private var ortEnv: OrtEnvironment = OrtEnvironment.getEnvironment()
    private lateinit var ortSession: OrtSession
    // Camera Variables
    private lateinit var viewBinding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraActivity: CameraActivity
    // UI Variables.
    private var imageView: ImageView? = null
    private var textView: TextView? = null
    // Everything else.
    private lateinit var modelProcessing: ModelProcessing

    override fun onCreate(savedInstanceState: Bundle?) {
        // Load OpenCV
        OpenCVLoader.initLocal()
        if(OpenCVLoader.initLocal()){
            Log.e("MyTag","OpenCV Loaded.")
        } else {
            Log.e("MyTag","OpenCV Not Loaded.")
        }
        // Initialize Model Processing and Camera Activity
        cameraActivity = CameraActivity()
        modelProcessing = ModelProcessing(resources)
        // Warm-up
        modelProcessing.warmupThreads()
        // Model Info
        modelProcessing.getModelInfo(1)
        modelProcessing.getModelInfo(2)
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewBinding = ActivityCameraBinding.inflate(layoutInflater)
        val btnCallCamera = findViewById<Button>(R.id.btnCallCamera)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        imageView = findViewById(R.id.imageView)

        // Button Listeners
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        btnCallCamera.setOnClickListener {
            if (allPermissionsGranted()) {
                // Call CameraActivity.kt
                val intent = Intent(this, CameraActivity::class.java)
                startCameraActivity.launch(intent)
            } else {
                requestPermissions()
            }
        }
        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    // Variables with lazy initialization
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            Log.d("Photo Picker", "Photo selected: $uri")
            val modelResults = modelProcessing.processImage(bitmap)
            displayImage(modelResults.detectionResult.outputBitmap)
            displayRecognitionResults(modelResults.recognitionResult.listOfStrings)
        } else {
            Log.d("Photo Picker", "No photo selected.")
        }
    }
    private val requestPermissionLauncher = registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()){
        permissions ->
        var permissionGranted = true
        permissions.entries.forEach {
            if (it.key in REQUIRED_PERMISSIONS && !it.value)
                permissionGranted = false
        }
        if (!permissionGranted) {
            Toast.makeText(baseContext,
                "Permission request denied",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(baseContext,
                "Camera open.",
                Toast.LENGTH_SHORT).show()
        }
    }
    private val startCameraActivity = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val data: Intent? = result.data
            val bitmapUri = Uri.parse(data?.getStringExtra("data"))
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, bitmapUri)
            imageView?.setImageBitmap(bitmap)
        }
    }
    // Functions
    private fun displayImage(bitmap: Bitmap?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageBitmap(bitmap)
    }
    private fun displayImageFromUri(imageUri: Uri?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageURI(imageUri)
    }
    private fun displayRecognitionResults(listOfStrings: MutableList<String>) {
        textView = findViewById(R.id.textView)
        textView!!.visibility = View.VISIBLE
        textView!!.text = "Recognition Results (Unordered):"
        for (string in listOfStrings) {
            textView!!.append("\n")
            textView!!.append(string)
        }
    }
    // Debug Functions
    private fun saveImage(bitmap: Bitmap, filename: String){
        val fileOutputStream = FileOutputStream(filename)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
    }
    // CameraX Functions
    private fun allPermissionsGranted() = REQUIRED_PERMISSIONS.all {
        ContextCompat.checkSelfPermission(
            baseContext, it) == PackageManager.PERMISSION_GRANTED
    }
    private fun requestPermissions() {
        requestPermissionLauncher.launch(REQUIRED_PERMISSIONS)
    }
    override fun onDestroy() {
        super.onDestroy()
        if (this::ortSession.isInitialized) {
            ortSession.close()
        }
        ortEnv.close()
        cameraExecutor.shutdown()
    }
    companion object {
        const val TAG = "CameraXApp"
        const val FILENAME_FORMAT = "yyyy-MM-dd-HH-mm-ss-SSS"
        private val REQUIRED_PERMISSIONS = arrayOf(android.Manifest.permission.CAMERA)
    }
}