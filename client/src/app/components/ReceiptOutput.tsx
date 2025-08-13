import React from "react";

export interface ReceiptItem {
  category: string;
  description: string;
  quantity: string;
  unit_price: string;
  total: string;
}

export interface ParsedData {
  "Company Name"?: string; // keys with spaces must be quoted
  Date?: string;
  Items?: ReceiptItem[];
  Subtotal?: string;
  "Sales Tax"?: string;
  Total?: string;
  raw_text?: string;
}

export interface OCRResponse {
  confidence: string;
  extraction_method: string;
  parsed_data: ParsedData;
  processing_time: number;
  raw_text: string;
}

type Props = {
  ocrOutput: OCRResponse;
};

const ReceiptOutput = ({
  ocrOutput: {
    confidence,
    extraction_method,
    parsed_data,
    processing_time,
    raw_text,
  },
}: Props) => {
  console.log(parsed_data);
  return (
    <div>
      <h2 className="text-lg font-semibold">OCR Output</h2>
      <div className="mt-4">
        <p>
          <strong>Total:</strong> {parsed_data.Total}
        </p>
        {/* <p>
          <strong>Raw Text:</strong>
        </p>
        <pre className="bg-gray-100 p-4 rounded">{raw_text}</pre> */}
      </div>
      <div className="mt-4">
        <h3 className="text-md font-semibold">Parsed Data</h3>
        <pre className="bg-gray-100 p-4 rounded">
          {JSON.stringify(parsed_data, null, 2)}
        </pre>
      </div>
    </div>
  );
};
