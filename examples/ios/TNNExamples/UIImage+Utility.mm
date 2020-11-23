// Tencent is pleased to support the open source community by making TNN available.
//
// Copyright (C) 2020 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

#import "UIImage+Utility.h"

#import <string>
#import <fstream>
#import <iostream>

namespace utility {
std::shared_ptr<char> UIImageGetData(UIImage *image) {
    int height = (int)CGImageGetHeight(image.CGImage);
    int width  = (int)CGImageGetWidth(image.CGImage);
    return UIImageGetData(image, height, width, 0);
}

std::shared_ptr<char> ImagetextGetData(NSString* path_, int height, int width) {
    auto data = std::shared_ptr<char>((char*)calloc(height * width * 4, 1), [](char *p) { free(p); });
    
    const std::string path = [path_ UTF8String];
    std::ifstream in(path);
    assert(in && in.good());
    
    for(int i=0; i<height * width * 4; ++i) {
        int pixel = 0;
        in >> pixel;
        data.get()[i] = static_cast<char>(pixel);
        //std::cout << i <<":\t" << pixel <<std::endl;
    }
    in.close();
    
    return data;
}

std::shared_ptr<char> UIImageGetData(UIImage *image, int height, int width, int gravity) {
    std::shared_ptr<char> data = nullptr;
    if (image == nil || image.CGImage == nil || height <= 0 || width <= 0) {
        return data;
    }

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    int cols                   = width;
    int rows                   = height;

    if (cols == 0 || rows == 0) {
        return data;
    }
    
    data = std::shared_ptr<char>((char*)calloc(rows * cols * 4, 1), [](char *p) { free(p); });

    CGContextRef contextRef =
        CGBitmapContextCreate(data.get(),                                             // Pointer to backing data
                              cols,                                                   // Width of bitmap
                              rows,                                                   // Height of bitmap
                              8,                                                      // Bits per component
                              cols * 4,                                               // Bytes per row
                              colorSpace,                                             // Colorspace
                              kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault); // Bitmap info flags

    CGContextSetInterpolationQuality(contextRef, kCGInterpolationHigh);
    
    int image_orig_height = (int)CGImageGetHeight(image.CGImage);
    int image_orig_width = (int)CGImageGetWidth(image.CGImage);
    
    float image_target_aspect = height/(float)(width + FLT_EPSILON);
    float image_orig_aspect = image_orig_height/(float)(image_orig_width + FLT_EPSILON);
    
    if (gravity == 1) {
        int offset_x = 0;
        int offset_y = 0;
        if (image_orig_aspect > image_target_aspect) {
            int object_aspect_width = height / image_orig_aspect;
            offset_x = (width - object_aspect_width) / 2;
        } else {
            int object_aspect_height = width * image_orig_aspect;
            offset_y = (height - object_aspect_height) / 2;
        }
        CGContextDrawImage(contextRef,
                           CGRectMake(offset_x, offset_y, cols-2*offset_x, rows - 2*offset_y),
                           image.CGImage);
    } else if (gravity == 2) {
        int offset_x = 0;
        int offset_y = 0;
        if (image_orig_aspect > image_target_aspect) {
            int object_aspect_height = image_orig_width * image_target_aspect;
            offset_y = (image_orig_height - object_aspect_height) / 2;
        } else {
            int object_aspect_width = image_orig_height / image_target_aspect;
            offset_x = (image_orig_width - object_aspect_width) / 2;
        }
        
        auto image_crop_ref = CGImageCreateWithImageInRect(image.CGImage,
                                                           CGRectMake(offset_x, offset_y, image_orig_width-2*offset_x, image_orig_height-2*offset_y));
        CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image_crop_ref);
        CGImageRelease(image_crop_ref);
    } else {
        CGContextDrawImage(contextRef, CGRectMake(0, 0, cols, rows), image.CGImage);
    }
    
    CGContextRelease(contextRef);
    CGColorSpaceRelease(colorSpace);
    
    int idx = 0;
    char* data_ptr = static_cast<char *>(data.get());
    for(int h=0; h<cols; ++h) {
        for(int w=0; w<rows; ++w) {
            printf("[%d %d %d %d]\n", static_cast<uint8_t>(data_ptr[idx]), static_cast<uint8_t>(data_ptr[idx+1]), static_cast<uint8_t>(data_ptr[idx+2]), static_cast<uint8_t>(data_ptr[idx+3]));
            idx += 4;
        }
    }
    
    CGImageRef cgimage = [image CGImage];
    CGImageAlphaInfo info = CGImageGetAlphaInfo(cgimage);
    if (info == kCGImageAlphaNone) {
        printf("alpha none");
    }
    if (info == kCGImageAlphaOnly) {
        printf("alpha only");
    }
    if (info == kCGImageAlphaFirst) {
        printf("alpha first");
    }
    if (info == kCGImageAlphaLast) {
        printf("alpha last");
    }
    if (info == kCGImageAlphaPremultipliedLast) {
        printf("alpha premultiplied last");
    }
    if (info == kCGImageAlphaPremultipliedFirst) {
        printf("alpha premultiplied first");
    }
    if (info == kCGImageAlphaNoneSkipLast) {
        printf("alpha none skip last");
    }
    if (info == kCGImageAlphaNoneSkipFirst) {
        printf("alpha none skip first");
    }
    
    return data;
}

UIImage * UIImageCrop(UIImage *image, CGRect rect) {
    CGImageRef imageRef = CGImageCreateWithImageInRect([image CGImage], rect);
    UIImage *croppedImage = [UIImage imageWithCGImage:imageRef];
    CGImageRelease(imageRef);
    return croppedImage;
}

UIImage *UIImageWithDataRGBA(void *image_data, int height, int width) {
    UIImage *image = nullptr;
    if (image_data == nil || height <= 0 || width <= 0) {
        return image;
    }

    NSData *data = [NSData dataWithBytes:image_data length:height * width * 4];

    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGDataProviderRef provider = CGDataProviderCreateWithCFData((CFDataRef)data);

    CGImageRef imageRef = CGImageCreate(width,                                                 // Width
                                        height,                                                // Height
                                        8,                                                     // Bits per component
                                        8 * 4,                                                 // Bits per pixel
                                        width * 4,                                             // Bytes per row
                                        colorSpace,                                            // Colorspace
                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault, // Bitmap info flags
                                        provider,                                              // CGDataProviderRef
                                        NULL,                                                  // Decode
                                        false,                                                 // Should interpolate
                                        kCGRenderingIntentDefault);                            // Intent

    image = [UIImage imageWithCGImage:imageRef];

    CGImageRelease(imageRef);
    CGDataProviderRelease(provider);
    CGColorSpaceRelease(colorSpace);
    return image;
}

UIImage *UIImageWithCVImageBuffRef(CVImageBufferRef imageBuffer) {
    CVPixelBufferLockBaseAddress(imageBuffer, 0);
    void *baseAddress = CVPixelBufferGetBaseAddress(imageBuffer);
    size_t bytesPerRow = CVPixelBufferGetBytesPerRow(imageBuffer);
    size_t width = CVPixelBufferGetWidth(imageBuffer);
    size_t height = CVPixelBufferGetHeight(imageBuffer);
    CGColorSpaceRef colorSpace = CGColorSpaceCreateDeviceRGB();
    CGContextRef context = CGBitmapContextCreate(baseAddress, width, height, 8,
                                                 bytesPerRow, colorSpace, kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGImageRef quartzImage = CGBitmapContextCreateImage(context);
    CVPixelBufferUnlockBaseAddress(imageBuffer,0);

    CGContextRelease(context);
    CGColorSpaceRelease(colorSpace);
    
    UIImage *image = [UIImage imageWithCGImage:quartzImage];
    CGImageRelease(quartzImage);
    
    return (image);
}

std::shared_ptr<char> CVImageBuffRefGetData(CVImageBufferRef image_buffer) {
    CGSize size = CVImageBufferGetDisplaySize(image_buffer);
    return CVImageBuffRefGetData(image_buffer, size.height, size.width);
}

std::shared_ptr<char> CVImageBuffRefGetData(CVImageBufferRef image_buffer, int target_height, int target_width) {
    std::shared_ptr<char> data = nullptr;
    if (image_buffer == nil){
        return data;
    }
    CGSize size = CVImageBufferGetDisplaySize(image_buffer);
    if (size.height <= 0 || size.width <= 0) {
        return data;
    }
    
    data = std::shared_ptr<char>(new char[target_height * target_width * 4], [](char* p) { delete[] p; });

    CGColorSpaceRef color_space = CGColorSpaceCreateDeviceRGB();
    
    
    CVPixelBufferLockBaseAddress(image_buffer, 0);
    void *base_address = CVPixelBufferGetBaseAddress(image_buffer);
    size_t bytes_per_row = CVPixelBufferGetBytesPerRow(image_buffer);
    size_t width = CVPixelBufferGetWidth(image_buffer);
    size_t height = CVPixelBufferGetHeight(image_buffer);
    CGContextRef context_orig = CGBitmapContextCreate(base_address, width, height, 8,
                                                 bytes_per_row, color_space,
                                                 kCGBitmapByteOrder32Little | kCGImageAlphaPremultipliedFirst);
    CGImageRef cgmage_orig = CGBitmapContextCreateImage(context_orig);
    
    {
        //resize
        CGContextRef context_target = CGBitmapContextCreate(data.get(),
                                                        target_width,
                                                        target_height,
                                                        8,
                                                        target_width * 4,
                                                        color_space,
                                                        kCGImageAlphaNoneSkipLast | kCGBitmapByteOrderDefault);

        CGContextSetInterpolationQuality(context_target, kCGInterpolationHigh);
        CGContextDrawImage(context_target, CGRectMake(0, 0, target_width, target_height), cgmage_orig);
        CGContextRelease(context_target);
    }
    
    CVPixelBufferUnlockBaseAddress(image_buffer,0);

    CGContextRelease(context_orig);
    CGColorSpaceRelease(color_space);
    CGImageRelease(cgmage_orig);
    
    return data;
}

}
