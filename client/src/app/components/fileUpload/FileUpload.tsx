"use client";

import { sendPDFReceiptForOCR } from "@/app/actions";
import React, { useCallback, useEffect } from "react";
import { useDropzone } from "react-dropzone";
import { OCRResponse } from "../ReceiptOutput";

type Props = {
  setOCRResponse: React.Dispatch<React.SetStateAction<OCRResponse | null>>;
  setIsLoading: React.Dispatch<React.SetStateAction<boolean>>;
};

export const FileUpload = ({ setOCRResponse, setIsLoading }: Props) => {
  const [fileName, setFileName] = React.useState<string | null>(null);
  const [fileContent, setFileContent] = React.useState<ArrayBuffer | null>(
    null
  );
  const [promptText, setPromptText] = React.useState<string>(
    "Upload Receipt (PDF or PNG)"
  );

  const borderColor = fileName ? "border-blue-500" : "border-gray-300";

  const onDrop = useCallback((acceptedFiles: File[]) => {
    acceptedFiles.forEach((file) => {
      const reader = new FileReader();
      setFileName(file.name);
      reader.onabort = () => console.log("file reading was aborted");
      reader.onerror = () => console.log("file reading has failed");
      reader.onload = () => {
        const binaryStr = reader.result;
        setFileContent(binaryStr as ArrayBuffer);
        console.log(binaryStr);
      };
      reader.readAsArrayBuffer(file);
    });
  }, []);

  useEffect(() => {
    if (!fileContent) return;

    let cancelled = false; // cleanup flag
    // optional: keep in a ref if you want cross-effect protection
    // reqIdRef.current = reqId;

    setOCRResponse(null); // clear old UI immediately
    setPromptText("Analyzing Receipt…");
    setIsLoading(true);

    (async () => {
      try {
        const result = await sendPDFReceiptForOCR(
          fileContent,
          fileName || "upload.pdf",
          fileName?.endsWith(".png") ? "png" : "pdf"
        );

        if (cancelled) return; // ignore if effect was re-run/unmounted

        setPromptText(`${fileName ?? "File"} analyzed successfully!`);
        setOCRResponse(result as OCRResponse);
      } catch (err) {
        if (!cancelled) console.error("Error during OCR:", err);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    })();

    return () => {
      cancelled = true; // prevents late updates
    };
    // Only rerun when inputs change (setter fns are stable; don’t include them)
  }, [fileContent, fileName, setIsLoading, setOCRResponse]);

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  return (
    <div className="p-10 m-20">
      <div
        className={`border-dashed border-2 ${borderColor} p-10 h-100 flex items-center justify-center cursor-pointer hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors rounded-lg`}
        {...getRootProps()}
      >
        <input {...getInputProps()} />
        <div>{promptText}</div>
      </div>
      {fileName && (
        <div className="mt-4 text-sm text-white">
          Selected file: <strong>{fileName}</strong>
          <button
            className="ml-4 text-blue-500 hover:underline"
            onClick={() => {
              setFileName(null);
              setFileContent(null);
              setOCRResponse(null);
              setPromptText("Upload Receipt (PDF or PNG)");
            }}
          >
            Clear File
          </button>
        </div>
      )}
    </div>
  );
};
