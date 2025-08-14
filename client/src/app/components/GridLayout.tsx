"use client";

import React, { useState } from "react";
import { FileUpload } from "./fileUpload/FileUpload";
import { OCRResponse, ReceiptOutput } from "./ReceiptOutput";

export const GridLayout = () => {
  const [ocrResponse, setOCRResponse] = useState<OCRResponse | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  return (
    <div className="bg-gray-50 dark:bg-gray-900">
      <div className="mx-auto max-w-2xl lg:max-w-7xl">
        <h2 className="text-base/7 font-semibold text-indigo-600 dark:text-indigo-400">
          Welcome to
        </h2>
        <p className="mt-2 max-w-lg text-4xl font-semibold tracking-tight text-pretty text-gray-900 sm:text-5xl dark:text-white">
          ReceiptIQ
        </p>
        <div className="flex gap-8 my-10">
          <div className="w-full overflow-hidden rounded-lg bg-white shadow-sm outline outline-black/5 max-lg:rounded-t-4xl lg:rounded-tl-4xl dark:bg-gray-800 dark:shadow-none dark:outline-white/15">
            <FileUpload
              setOCRResponse={setOCRResponse}
              setIsLoading={setIsLoading}
            />
          </div>
          <div className="flex w-full overflow-hidden rounded-lg bg-white shadow-sm outline outline-black/5 max-lg:rounded-b-4xl lg:rounded-br-4xl dark:bg-gray-800 dark:shadow-none dark:outline-white/15">
            <ReceiptOutput ocrOutput={ocrResponse} isLoading={isLoading} />
          </div>
        </div>
      </div>
    </div>
  );
};
