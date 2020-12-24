package com.tencent.tnn.demo.StreamObjectDetector;

import android.hardware.Camera;
import android.os.Bundle;
import android.util.Log;
import android.view.KeyEvent;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.widget.Button;
import android.widget.CompoundButton;
import android.widget.TextView;
import android.widget.ToggleButton;

import com.tencent.tnn.demo.ObjectDetector;
<<<<<<< HEAD
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
=======
import com.tencent.tnn.demo.FpsCounter;
import com.tencent.tnn.demo.FileUtils;
import com.tencent.tnn.demo.Helper;
import com.tencent.tnn.demo.ObjectInfo;
>>>>>>> origin/feature_demo_blazepose
import com.tencent.tnn.demo.R;
import com.tencent.tnn.demo.common.component.CameraSetting;
import com.tencent.tnn.demo.common.component.DrawView;
import com.tencent.tnn.demo.common.fragment.BaseFragment;
import com.tencent.tnn.demo.common.sufaceHolder.DemoSurfaceHolder;

import java.io.IOException;


public class StreamObjectDetectFragment extends BaseFragment {

    private final static String TAG = StreamObjectDetectFragment.class.getSimpleName();

    /**********************************     Define    **********************************/

    private SurfaceView mPreview;

    private DrawView mDrawView;
    private int mCameraWidth;
    private int mCameraHeight;

    Camera mOpenedCamera;
    int mOpenedCameraId = 0;
    DemoSurfaceHolder mDemoSurfaceHolder = null;

    private static final int NET_H_INPUT = 448;
    private static final int NET_W_INPUT = 640;

    int mCameraFacing = -1;
    int mRotate = -1;
    SurfaceHolder mSurfaceHolder;

    private ObjectDetector mObjectDetector = new ObjectDetector();
    private boolean mIsDetectingObject = false;
<<<<<<< HEAD
=======
    private FpsCounter mFpsCounter = new FpsCounter();
    private boolean mIsCountFps = false;
>>>>>>> origin/feature_demo_blazepose

    private ToggleButton mGPUSwitch;
    private boolean mUseGPU = false;
    //add for npu
<<<<<<< HEAD
    private ToggleButton mNPUswitch;
    private boolean mUseNPU = false;
    private TextView NpuTextView;
=======
    private ToggleButton mHuaweiNPUswitch;
    private boolean mUseHuaweiNpu = false;
    private TextView HuaweiNpuTextView;

    private boolean mDeviceSwiched = false;
>>>>>>> origin/feature_demo_blazepose

    /**********************************     Get Preview Advised    **********************************/

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.d(TAG, "onCreate");
        super.onCreate(savedInstanceState);
        System.loadLibrary("tnn_wrapper");
        //start SurfaceHolder
        mDemoSurfaceHolder = new DemoSurfaceHolder(this);
        String modelPath = initModel();
        NpuEnable = mObjectDetector.checkNpu(modelPath);
    }

    private String initModel()
    {

        String targetDir =  getActivity().getFilesDir().getAbsolutePath();

        //copy detect model to sdcard
        String[] modelPathsDetector = {
                "yolov5s.tnnmodel",
                "yolov5s-permute.tnnproto",
        };

        for (int i = 0; i < modelPathsDetector.length; i++) {
            String modelFilePath = modelPathsDetector[i];
            String interModelFilePath = targetDir + "/" + modelFilePath ;
            FileUtils.copyAsset(getActivity().getAssets(), "yolov5/"+modelFilePath, interModelFilePath);
        }
        return targetDir;
    }

    @Override
    public void onClick(View view) {
        int i = view.getId();
        if (i == R.id.back_rl) {
            clickBack();
        }
    }
    private void restartCamera()
    {
        closeCamera();
        openCamera(mCameraFacing);
        startPreview(mSurfaceHolder);
    }
    private void onSwichGPU(boolean b)
    {
<<<<<<< HEAD
        if (b && mNPUswitch.isChecked()) {
            mNPUswitch.setChecked(false);
            mUseNPU = false;
=======
        if (b && mHuaweiNPUswitch.isChecked()) {
            mHuaweiNPUswitch.setChecked(false);
            mUseHuaweiNpu = false;
>>>>>>> origin/feature_demo_blazepose
        }
        mUseGPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
<<<<<<< HEAD
        restartCamera();
=======
        mDeviceSwiched = true;
>>>>>>> origin/feature_demo_blazepose
    }

    private void onSwichNPU(boolean b)
    {
        if (b && mGPUSwitch.isChecked()) {
            mGPUSwitch.setChecked(false);
            mUseGPU = false;
        }
<<<<<<< HEAD
        mUseNPU = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        restartCamera();
=======
        mUseHuaweiNpu = b;
        TextView result_view = (TextView)$(R.id.result);
        result_view.setText("");
        mDeviceSwiched = true;
>>>>>>> origin/feature_demo_blazepose
    }

    private void clickBack() {
        if (getActivity() != null) {
            (getActivity()).finish();
        }
    }

    @Override
    public void setFragmentView() {
        Log.d(TAG, "setFragmentView");
<<<<<<< HEAD
        setView(R.layout.fragment_streamfacedetector);
=======
        setView(R.layout.fragment_stream_detector);
>>>>>>> origin/feature_demo_blazepose
        setTitleGone();
        $$(R.id.gpu_switch);
        $$(R.id.back_rl);
        mGPUSwitch = $(R.id.gpu_switch);
        mGPUSwitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwichGPU(b);
            }
        });

        $$(R.id.npu_switch);
<<<<<<< HEAD
        mNPUswitch = $(R.id.npu_switch);
        mNPUswitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
=======
        mHuaweiNPUswitch = $(R.id.npu_switch);
        mHuaweiNPUswitch.setOnCheckedChangeListener(new CompoundButton.OnCheckedChangeListener() {
>>>>>>> origin/feature_demo_blazepose
            @Override
            public void onCheckedChanged(CompoundButton compoundButton, boolean b) {
                onSwichNPU(b);
            }
        });
<<<<<<< HEAD
        NpuTextView = $(R.id.npu_text);
        if (!NpuEnable) {
            NpuTextView.setVisibility(View.INVISIBLE);
            mNPUswitch.setVisibility(View.INVISIBLE);
=======
        HuaweiNpuTextView = $(R.id.npu_text);
        if (!NpuEnable) {
            HuaweiNpuTextView.setVisibility(View.INVISIBLE);
            mHuaweiNPUswitch.setVisibility(View.INVISIBLE);
>>>>>>> origin/feature_demo_blazepose
        }
        init();
    }


    private void init() {
        mPreview = $(R.id.live_detection_preview);

        Button btnSwitchCamera = $(R.id.switch_camera);
        btnSwitchCamera.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View v) {
                closeCamera();
                if (mCameraFacing == Camera.CameraInfo.CAMERA_FACING_FRONT) {
                    openCamera(Camera.CameraInfo.CAMERA_FACING_BACK);
                }
                else {
                    openCamera(Camera.CameraInfo.CAMERA_FACING_FRONT);
                }
                startPreview(mSurfaceHolder);
            }
        });


        mDrawView = (DrawView) $(R.id.drawView);


    }

    @Override
    public void onStart() {
        Log.d(TAG, "onStart");
        super.onStart();
        if (null != mDemoSurfaceHolder) {
            SurfaceHolder holder = mPreview.getHolder();
            holder.setKeepScreenOn(true);
            mDemoSurfaceHolder.setSurfaceHolder(holder);
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        Log.d(TAG, "onResume");

        getFocus();
        preview();
    }

    @Override
    public void onPause() {
        Log.d(TAG, "onPause");
        super.onPause();
    }

    @Override
    public void onStop() {
        Log.i(TAG, "onStop");
        super.onStop();
    }


    @Override
    public void onDestroy() {
        super.onDestroy();
        Log.i(TAG, "onDestroy");
    }

    private void preview() {
        Log.i(TAG, "preview");

    }

    private void getFocus() {
        getView().setFocusableInTouchMode(true);
        getView().requestFocus();
        getView().setOnKeyListener(new View.OnKeyListener() {
            @Override
            public boolean onKey(View v, int keyCode, KeyEvent event) {
            if (event.getAction() == KeyEvent.ACTION_UP && keyCode == KeyEvent.KEYCODE_BACK) {
                clickBack();
                return true;
            }
            return false;
            }
        });
    }

/**********************************     Camera    **********************************/


    public void openCamera() {
<<<<<<< HEAD
        openCamera(Camera.CameraInfo.CAMERA_FACING_FRONT);
=======
        openCamera(Camera.CameraInfo.CAMERA_FACING_BACK);
>>>>>>> origin/feature_demo_blazepose
    }

    private void openCamera(int cameraFacing) {
        mIsDetectingObject = true;
        mCameraFacing = cameraFacing;
        try {
            int numberOfCameras = Camera.getNumberOfCameras();
            if (numberOfCameras < 1) {
                Log.e(TAG, "no camera device found");
            } else if (1 == numberOfCameras) {
                mOpenedCamera = Camera.open(0);
                mOpenedCameraId = 0;
            } else {
                Camera.CameraInfo cameraInfo = new Camera.CameraInfo();
                for (int i = 0; i < numberOfCameras; i++) {
                    Camera.getCameraInfo(i, cameraInfo);
                    if (cameraInfo.facing == cameraFacing) {
                        mOpenedCamera = Camera.open(i);
                        mOpenedCameraId = i;
                        break;
                    }
                }
            }
            if (mOpenedCamera == null) {
//                popTip("can't find camera","");
                Log.e(TAG, "can't find camera");
            }
            else {

                int r = CameraSetting.initCamera(getActivity().getApplicationContext(),mOpenedCamera,mOpenedCameraId);
                if (r == 0) {
                    //设置摄像头朝向
                    CameraSetting.setCameraFacing(cameraFacing);

                    Camera.Parameters parameters = mOpenedCamera.getParameters();
                    mRotate = CameraSetting.getRotate(getActivity().getApplicationContext(), mOpenedCameraId, mCameraFacing);
                    mCameraWidth = parameters.getPreviewSize().width;
                    mCameraHeight = parameters.getPreviewSize().height;
                    String modelPath = initModel();
                    int device = 0;
<<<<<<< HEAD
                    if (mUseNPU) {
=======
                    if (mUseHuaweiNpu) {
>>>>>>> origin/feature_demo_blazepose
                        device = 2;
                    } else if (mUseGPU) {
                        device = 1;
                    }
<<<<<<< HEAD
                    int ret = mObjectDetector.init(modelPath, NET_H_INPUT, NET_W_INPUT, 0.975f, 0.23f, 1, device);
=======
                    int ret = mObjectDetector.init(modelPath, NET_W_INPUT, NET_H_INPUT, 0.7f, 0.3f, -1, device);
>>>>>>> origin/feature_demo_blazepose
                    if (ret == 0) {
                        mIsDetectingObject = true;
                    } else {
                        mIsDetectingObject = false;
                        Log.e(TAG, "Face detector init failed " + ret);
                    }
<<<<<<< HEAD
=======

                    ret = mFpsCounter.init();
                    if (ret == 0) {
                        mIsCountFps = true;
                    } else {
                        mIsCountFps = false;
                        Log.e(TAG, "Fps Counter init failed " + ret);
                    }
>>>>>>> origin/feature_demo_blazepose
                } else {
                    Log.e(TAG, "Failed to init camera");
                }
            }
        }
        catch (Exception e) {
            Log.e(TAG, "open camera failed:" + e.getLocalizedMessage());
        }
    }

    public void startPreview(SurfaceHolder surfaceHolder) {
        try {
            if (null != mOpenedCamera) {
                Log.i(TAG, "start preview, is previewing");
                mOpenedCamera.setPreviewCallback(new Camera.PreviewCallback() {
                    @Override
                    public void onPreviewFrame(byte[] data, Camera camera) {
                        if (mIsDetectingObject) {
                            Camera.Parameters mCameraParameters = camera.getParameters();
<<<<<<< HEAD
                            ObjectDetector.ObjectInfo[] objectInfoList = mObjectDetector.detectFromStream(data, mCameraParameters.getPreviewSize().width, mCameraParameters.getPreviewSize().height, mRotate);
=======
                            ObjectInfo[] objectInfoList;
                            // reinit
                            if (mDeviceSwiched) {
                                String modelPath = getActivity().getFilesDir().getAbsolutePath();
                                int device = 0;
                                if (mUseHuaweiNpu) {
                                    device = 2;
                                } else if (mUseGPU) {
                                    device = 1;
                                }
                                int ret = mObjectDetector.init(modelPath, NET_W_INPUT, NET_H_INPUT, 0.7f, 0.3f, -1, device);
                                if (ret == 0) {
                                    mIsDetectingObject = true;
                                    mFpsCounter.init();
                                } else {
                                    mIsDetectingObject = false;
                                    Log.e(TAG, "Face detector init failed " + ret);
                                }
                                mDeviceSwiched = false;
                            }
                            if (mIsCountFps) {
                                mFpsCounter.begin("ObjectDetect");
                            }
                            objectInfoList = mObjectDetector.detectFromStream(data, mCameraParameters.getPreviewSize().width, mCameraParameters.getPreviewSize().height, mDrawView.getWidth(), mDrawView.getHeight(), mRotate);
                            if (mIsCountFps) {
                                mFpsCounter.end("ObjectDetect");
                                double fps = mFpsCounter.getFps("ObjectDetect");
                                String monitorResult = "device: ";
                                if (mUseGPU) {
                                    monitorResult += "opencl\n";
                                } else if (mUseHuaweiNpu) {
                                    monitorResult += "huawei_npu\n";
                                } else {
                                    monitorResult += "arm\n";
                                }
                                monitorResult += "fps: " + String.format("%.02f", fps);
                                TextView monitor_result_view = (TextView)$(R.id.monitor_result);
                                monitor_result_view.setText(monitorResult);
                            }
>>>>>>> origin/feature_demo_blazepose
                            Log.i(TAG, "detect from stream ret " + objectInfoList);
                            int objectCount = 0;
                            if (objectInfoList != null) {
                                objectCount = objectInfoList.length;
                            }
<<<<<<< HEAD
                            mDrawView.addObjectRect(objectInfoList, mCameraParameters.getPreviewSize().height, mCameraParameters.getPreviewSize().width);

                            String result = "object count: " + objectCount + " " + Helper.getBenchResult();
                            TextView result_view = (TextView)$(R.id.result);
                            result_view.setText(result);
=======
                            mDrawView.addObjectRect(objectInfoList,  ObjectDetector.label_list);
>>>>>>> origin/feature_demo_blazepose
                        }
                        else {
                            Log.i(TAG,"No object");
                        }
                    }
                });
                mOpenedCamera.setPreviewDisplay(surfaceHolder);
                mOpenedCamera.startPreview();
                mSurfaceHolder = surfaceHolder;
            }
        } catch (IOException e) {
            e.printStackTrace();
        } catch (RuntimeException e) {
            e.printStackTrace();
        }
    }

    public void closeCamera() {
        Log.i(TAG, "closeCamera");
        mIsDetectingObject = false;
        if (mOpenedCamera != null) {
            try {
                mOpenedCamera.stopPreview();
                mOpenedCamera.setPreviewCallback(null);
                Log.i(TAG, "stop preview, not previewing");
            } catch (Exception e) {
                e.printStackTrace();
                Log.i(TAG, "Error setting camera preview: " + e.toString());
            }
            try {
                mOpenedCamera.release();
                mOpenedCamera = null;
            } catch (Exception e) {
                e.printStackTrace();
                Log.i(TAG, "Error setting camera preview: " + e.toString());
            } finally {
                mOpenedCamera = null;
            }
        }
        mObjectDetector.deinit();
    }

}
