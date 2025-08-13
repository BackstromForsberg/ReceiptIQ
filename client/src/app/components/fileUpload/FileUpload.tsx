"use client";

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

export const FileUpload = () => {
  const [fileName, setFileName] = React.useState<string | null>(null);
  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();
      setFileName(file.name);
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
    <div className="p-10 m-20">
      <div className="border-dashed border-2 p-10" {...getRootProps()}>
        <input {...getInputProps()} />
        <div className="text-center m-0-auto">Upload Receipt PDF</div>
      </div>
      {fileName && (
        <div className="mt-4 text-sm text-gray-600">
          Selected file: <strong>{fileName}</strong>
        </div>
      )}
    </div>
  );
};
