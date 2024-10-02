package com.example.thesisocr

import android.annotation.SuppressLint
import android.app.Activity
import android.content.ContentValues
import android.content.Intent
import android.content.pm.ActivityInfo
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.icu.text.SimpleDateFormat
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.provider.MediaStore
import android.util.Log
import android.util.Rational
import android.view.MotionEvent
import android.view.TextureView
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import android.widget.Toast
import androidx.annotation.RequiresApi
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.camera2.Camera2Config
import androidx.camera.core.AspectRatio
import androidx.camera.core.CameraSelector
import androidx.camera.core.CameraX
import androidx.camera.core.CameraXConfig
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCapture.CaptureMode
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.core.UseCaseGroup
import androidx.camera.core.ViewPort
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.core.content.ContextCompat
import com.example.thesisocr.databinding.ActivityCameraBinding
import java.io.File
import java.io.FileOutputStream
import java.util.Locale
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

class CameraActivity: AppCompatActivity(), CameraXConfig.Provider{
    data class CameraResult (
        var imageResult: Bitmap
    )
    // Camera Variables
    private lateinit var viewBinding: ActivityCameraBinding
    private lateinit var cameraExecutor: ExecutorService
    private var imageCapture: ImageCapture? = null
    private lateinit var cameraControl: androidx.camera.core.CameraControl
    // UI Variables.
    private var imageView: ImageView? = null
    private var textView: TextView? = null
    private var previewView: PreviewView? = null

    @SuppressLint("ClickableViewAccessibility")
    @RequiresApi(Build.VERSION_CODES.R)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        viewBinding = ActivityCameraBinding.inflate(layoutInflater)
        setContentView(viewBinding.root)
        // Lock the screen orientation to landscape.
        requestedOrientation = ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE

        startCamera()

        // Buttons
        val btnTakePhoto = findViewById<Button>(R.id.btnTakePhoto)
        val btnExitCamera = findViewById<Button>(R.id.btnExitCamera)
        val previewView = findViewById<PreviewView>(R.id.viewFinder)
        previewView.scaleType = PreviewView.ScaleType.FIT_CENTER

        previewView.setOnTouchListener { _, event ->
            if (event.action == MotionEvent.ACTION_DOWN) {
                val factory = previewView.meteringPointFactory
                val point = factory.createPoint(event.x, event.y)
                val action = FocusMeteringAction.Builder(point).build()
                cameraControl.startFocusAndMetering(action)
            }
            return@setOnTouchListener true
        }

        btnTakePhoto.setOnClickListener {
            takePhoto()
            Toast.makeText(this, "Photo taken. Wait for processing.", Toast.LENGTH_SHORT).show()
            Toast.makeText(this, "You can let go now.", Toast.LENGTH_SHORT).show()
        }
        btnExitCamera.setOnClickListener {
            // Return to the previous activity.
            finish()
        }

        cameraExecutor = Executors.newSingleThreadExecutor()
    }
    private fun takePhoto() {
        val imageCapture = imageCapture ?: return
        imageCapture.takePicture(
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageCapturedCallback() {
                override fun onError(exception: ImageCaptureException) {
                    Log.e(MainActivity.TAG, "Photo capture failed: ${exception.message}", exception)
                }
                override fun onCaptureSuccess(image: ImageProxy) {
                    val bitmap = imageProxyToBitmap(image)
                    imageView?.setImageBitmap(bitmap)
                    image.close()

                    val bitmapUri = saveBitmapToFile(bitmap)
                    val resultIntent = Intent().apply {
                        putExtra("data", bitmapUri.toString())
                    }
                    setResult(Activity.RESULT_OK, resultIntent)
                    finish()
                }
            }
        )

    }
    @SuppressLint("ClickableViewAccessibility")
    private fun startCamera(){
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            val preview = Preview.Builder()
                .build().also {
                it.setSurfaceProvider(viewBinding.viewFinder.surfaceProvider)
            }
            imageCapture = ImageCapture.Builder()
                //.setFlashMode(ImageCapture.FLASH_MODE_ON)
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            try {
                cameraProvider.unbindAll()
                val camera = cameraProvider.bindToLifecycle(this, cameraSelector, preview, imageCapture)
                cameraControl = camera.cameraControl
                cameraControl.setZoomRatio(2.0f)
            } catch (e: Exception) {
                Log.e(MainActivity.TAG, "Use case binding failed", e)
            }
        }, ContextCompat.getMainExecutor(this))
    }
    private fun imageProxyToBitmap(image: ImageProxy): Bitmap {
        val buffer = image.planes[0].buffer
        buffer.rewind()
        val bytes = ByteArray(buffer.capacity())
        buffer.get(bytes)
        return BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
    }
    private fun saveBitmapToFile(bitmap: Bitmap): Uri {
        // Create a file in the cache directory
        val file = File(externalCacheDir, "${System.currentTimeMillis()}.jpg")
        val fos = FileOutputStream(file)
        bitmap.compress(Bitmap.CompressFormat.JPEG, 100, fos)
        fos.close()
        return Uri.fromFile(file)
    }

    override fun getCameraXConfig(): CameraXConfig = CameraXConfig.Builder.fromConfig(Camera2Config.defaultConfig()).build()
}