"use client";

import React, { useCallback } from "react";
import { useDropzone } from "react-dropzone";

export const FileUpload = () => {
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [fileContent, setFileContent] = React.useState<ArrayBuffer | null>(
    null
  );
  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();
      setFileName(file.name);
      reader.onabort = () => console.log("file reading was aborted");
      reader.onerror = () => console.log("file reading has failed");
      reader.onload = () => {
        // Do whatever you want with the file contents
        const binaryStr = reader.result;
        setFileContent(binaryStr as ArrayBuffer);
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
          <button
            className="ml-4 text-blue-500 hover:underline"
            onClick={() => {
              setFileName(null);
              setFileContent(null);
            }}
          >
            Clear File
          </button>
        </div>
      )}
    </div>
  );
};
