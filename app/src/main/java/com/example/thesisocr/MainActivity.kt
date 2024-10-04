package com.example.thesisocr

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.content.Context
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.icu.text.SimpleDateFormat
import android.net.Uri
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
import androidx.camera.core.ImageCapture
import androidx.core.content.ContextCompat
import com.example.thesisocr.databinding.ActivityCameraBinding
import org.opencv.android.OpenCVLoader
import java.io.FileOutputStream
import java.io.OutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors
import com.google.gson.Gson
import java.io.File

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
    private lateinit var textView: TextView
    private lateinit var inferenceView: TextView
    // Everything else.
    private lateinit var modelProcessing: ModelProcessing
    private lateinit var modelResults: ModelProcessing.ModelResults

    private var parallelDetection = false
    private var runCount = 0
    private var cameraUsed = false
    private lateinit var cameraInputBitmap: Bitmap

    override fun onCreate(savedInstanceState: Bundle?) {
        // Get permissions.
        allPermissionsGranted()
        requestPermissions()
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
        Toast.makeText(baseContext,
            "Initializing models.",
            Toast.LENGTH_SHORT).show()
        modelProcessing.warmupThreads()
        // Model Info
        modelProcessing.debugGetModelInfo(1)
        modelProcessing.debugGetModelInfo(2)
        // Android Application Stuff
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)
        viewBinding = ActivityCameraBinding.inflate(layoutInflater)
        val textView = findViewById<TextView>(R.id.textView)
        textView.movementMethod = android.text.method.ScrollingMovementMethod()
        // Button declarations
        val btnCallCamera = findViewById<Button>(R.id.btnCallCamera)
        val btnSelectImage = findViewById<Button>(R.id.btnSelectImage)
        val btnDebugSave = findViewById<Button>(R.id.debugSave)
        val btnDebugProcess = findViewById<Button>(R.id.debugProcess)
        imageView = findViewById(R.id.imageView)

        // Button Listeners
        // Select Image Button
        btnSelectImage.setOnClickListener {
            pickMedia.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
        }
        // Call Camera Button
        btnCallCamera.setOnClickListener {
            startCameraActivity.launch(Intent(this, CameraActivity::class.java))
        }
        // Debug Save Button
        btnDebugSave.setOnClickListener {
            debugSaveImages(cameraUsed)
        }
        // Process Mode Button
        btnDebugProcess.setOnClickListener {
            // Toggle parallel detection.
            parallelDetection = !parallelDetection
            if (parallelDetection){
                Toast.makeText(baseContext,
                    "Parallel Detection: ON",
                    Toast.LENGTH_SHORT).show()
                btnDebugProcess.text = getString(R.string.debug_parallel)
            } else {
                Toast.makeText(baseContext,
                    "Parallel Detection: OFF",
                    Toast.LENGTH_SHORT).show()
                btnDebugProcess.text = getString(R.string.debug_serial)
            }
            Log.d("Parallel Detection", "Status: $parallelDetection")
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    // Variables with lazy initialization
    private val pickMedia = registerForActivityResult(ActivityResultContracts.PickVisualMedia()) { uri ->
        displayImageFromUri(uri)
        if (uri != null){
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, uri)
            // Process the image.
            processBitmap(bitmap, parallelDetection)
        } else {
            Log.d("Photo Picker", "No photo selected.")
        }
        cameraUsed = false
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
                "Camera Access Denied",
                Toast.LENGTH_SHORT).show()
        }
    }
    private val startCameraActivity = registerForActivityResult(ActivityResultContracts.StartActivityForResult()) { result ->
        if (result.resultCode == RESULT_OK) {
            val data: Intent? = result.data
            val bitmapUri = Uri.parse(data?.getStringExtra("data"))
            val bitmap = MediaStore.Images.Media.getBitmap(this.contentResolver, bitmapUri)
            // Process the image.
            cameraUsed = true
            cameraInputBitmap = bitmap
            processBitmap(bitmap, parallelDetection)
        }
    }
    // Functions
    private fun processBitmap(bitmap: Bitmap, parallelDetection: Boolean = true): Boolean {
        // Process the image.
        modelResults = modelProcessing.processImage(bitmap, parallelDetection)
        // Check if recognition result is null.
        if (modelResults.recognitionResult == null){
            Toast.makeText(baseContext,
                "Image not good. Please try again.",
                Toast.LENGTH_SHORT).show()
            textView = findViewById(R.id.textView)
            textView.visibility = View.VISIBLE
            textView.text = "Image not good. Please try again."
            return false
        } else {
            // Display the image and recognition results.
            displayImage(modelResults.preProcessedImage)
            // Check if modelResults.recognitionResult is null.
            displayRecognitionResults(modelResults.recognitionResult)
            // Display inference times to the user.
            displayInferenceTime(
                modelResults.detectionResult.inferenceTime.inWholeMilliseconds,
                modelResults.recognitionResult!!.inferenceTime.inWholeMilliseconds
            )
            runCount++
            return true
        }
    }
    private fun displayImage(bitmap: Bitmap?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageBitmap(bitmap)
    }
    private fun displayImageFromUri(imageUri: Uri?) {
        imageView!!.visibility = View.VISIBLE
        imageView!!.setImageURI(imageUri)
    }
    private fun displayRecognitionResults(recognitionResult: PaddleRecognition.TextResult?) {
        val listOfStrings = recognitionResult?.listOfStrings
        textView = findViewById(R.id.textView)
        textView.visibility = View.VISIBLE
        textView.text = "Recognition Results:"
        if (listOfStrings != null) {
            for (string in listOfStrings) {
                textView.append("\n")
                textView.append(string)
            }
        }
    }
    private fun displayInferenceTime(detectionInferenceTime: Long, recognitionInferenceTime: Long) {
        inferenceView = findViewById(R.id.inferenceView)
        inferenceView.visibility = View.VISIBLE
        inferenceView.text = "Detection inference time: $detectionInferenceTime ms"
        inferenceView.append("\nRecognition inference time: $recognitionInferenceTime ms")
    }
    // Debug Functions
    private fun debugSaveImages(cameraUsed: Boolean = false){
        // Grab bitmap contents of modelProcessing.
        val detectionBitmapMask = modelResults.detectionResult.outputMask
        val dateFormat = SimpleDateFormat("yyyy-MM-dd-HH-mm-ss", Locale.US)
        val filenameAppendix = dateFormat.format(System.currentTimeMillis())
        // Save the images.
        val yeetLocation = if (cameraUsed) {
            "/storage/emulated/0/Documents/ThesisOCR/Camera"
        } else {
            "/storage/emulated/0/Documents/ThesisOCR/Gallery"
        }
        val isDirectoryCreated = createDirectory(yeetLocation)
        if (!isDirectoryCreated){
            Toast.makeText(baseContext,
                "Directory not created.",
                Toast.LENGTH_SHORT).show()
            return
        }
        val detectionMaskBool = saveImage(detectionBitmapMask,
            "$yeetLocation/detection_mask_$filenameAppendix.jpg"
        )
        val cameraInputBool = if (cameraUsed) {
            saveImage(cameraInputBitmap,
                "$yeetLocation/camera_input_$filenameAppendix.jpg"
            )
        } else {
            false
        }
        saveStringListAsJson(modelResults.recognitionResult!!.listOfStrings,
            "$yeetLocation/recognition_output_$filenameAppendix.json"
        )
        saveBoundingBoxCoordinatesAsJson(modelResults.detectionResult.boundingBoxList,
            "$yeetLocation/detection_output_$filenameAppendix.json"
        )
        val combinedMap = createCombinedDictionary(
            modelResults.detectionResult.boundingBoxList,
            modelResults.recognitionResult!!.listOfStrings
        )
        appendCombinedListToJson(
            filenameAppendix,
            "$yeetLocation/combined_output.json",
            listOf(combinedMap)
        )
        // Check if at least one image is saved.
        if (detectionMaskBool || cameraInputBool){
            Toast.makeText(baseContext,
                "Image/s saved to $yeetLocation",
                Toast.LENGTH_SHORT).show()
        } else {
            Toast.makeText(baseContext,
                "Image/s not saved.",
                Toast.LENGTH_SHORT).show()
        }
    }
    private fun createDirectory(directoryName: String): Boolean {
        val directory = File(directoryName)
        return if (!directory.exists()){
            directory.mkdirs()
        } else {
            true
        }
    }
    private fun saveImage(bitmap: Bitmap, filename: String): Boolean {
        val fileOutputStream = FileOutputStream(filename)
        val saveBool = bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fileOutputStream)
        fileOutputStream.close()
        return saveBool
    }
    private fun saveStringListAsJson(inputList: List<String>, filename: String): Boolean {
        val gson = Gson()
        val jsonString = gson.toJson(inputList)
        File(filename).writeText(jsonString)
        return true
    }
    private fun saveBoundingBoxCoordinatesAsJson(inputList: List<PaddleDetector.BoundingBox>, filename: String): Boolean {
        val gson = Gson()
        val jsonString = gson.toJson(inputList)
        File(filename).writeText(jsonString)
        return true
    }
    private fun createCombinedDictionary
                (inputCoordinateList: List<PaddleDetector.BoundingBox>,
                 inputStringList: List<String>): Map<String, Any>
    {
        // Key:Values are "Bounding Box Coordinates":List<BoundingBox>, "Recognition Output":List<String>.
        val combinedDictionary = mutableMapOf<String, Any>()
        combinedDictionary["Bounding Box Coordinates"] = inputCoordinateList
        combinedDictionary["Recognition Output"] = inputStringList
        return combinedDictionary
    }
    private fun appendCombinedListToJson(inputKey: String, filename: String, inputMat: List<Map<String, Any>>): Boolean {
        // Append the format: inputKey:inputMat to the filename.
        // Create a new JSON file if it doesn't exist, otherwise append to the existing file.
        val gson = Gson()
        val jsonString = gson.toJson(mapOf(inputKey to inputMat))
        // Check if the file exists.
        if (!File(filename).exists()){
            File(filename).writeText(jsonString)
        } else {
            File(filename).appendText(jsonString)
        }
        return true
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