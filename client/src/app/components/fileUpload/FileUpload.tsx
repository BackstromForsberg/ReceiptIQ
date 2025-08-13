"use client";

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

export const FileUpload = () => {
  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();

      reader.onabort = () => console.log("file reading was aborted");
      reader.onerror = () => console.log("file reading has failed");
      reader.onload = () => {
        // Do whatever you want with the file contents
        const binaryStr = reader.result;
        console.log(binaryStr);
      };
      reader.readAsArrayBuffer(file);
    });
  }, []);
  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div className="p-10 m-20 border-dashed border-2" {...getRootProps()}>
      <input {...getInputProps()} />
      <div className="text-center">Upload Receipt PDF</div>
    </div>
  );
};
