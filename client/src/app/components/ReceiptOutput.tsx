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
  ocrOutput: OCRResponse | null;
};

export const ReceiptOutput = ({ ocrOutput }: Props) => {
  if (!ocrOutput) {
    return (
      <div className="m-10 w-200">Upload a PDF Receipt to see output.</div>
    );
  }

  const { confidence, extraction_method, parsed_data, processing_time } =
    ocrOutput;

  console.log(parsed_data);
  return (
    <div className="m-10">
      <h2 className="text-lg font-semibold">
        {parsed_data["Company Name"]} - Receipt OCR Output
      </h2>
      <h5>Date: {parsed_data.Date}</h5>
      <div className="mt-4">
        <h2 className="text-lg font-semibold">High Level Financials</h2>
        <p>
          <strong>Subtotal:</strong> {parsed_data.Subtotal}
        </p>
        <p>
          <strong>Tax:</strong> {parsed_data["Sales Tax"]}
        </p>
        <p>
          <strong>Total:</strong> {parsed_data.Total}
        </p>
      </div>
      <div className="mt-4">
        <h2 className="text-lg font-semibold">Scan Results</h2>
        <p>
          <strong>Confidence:</strong> {confidence}
        </p>
        <p>
          <strong>Extraction Method:</strong> {extraction_method}
        </p>
        <p>
          <strong>Processing Time:</strong> {processing_time} seconds
        </p>
        <div>
          {parsed_data.Items && parsed_data.Items.length > 0 ? (
            <table className="mt-4 w-full table-auto border-collapse">
              <thead>
                <tr>
                  <th className="border px-4 py-2">Description</th>
                  <th className="border px-4 py-2">Quantity</th>
                  <th className="border px-4 py-2">Unit Price</th>
                  <th className="border px-4 py-2">Total</th>
                </tr>
              </thead>
              <tbody>
                {parsed_data.Items.map((item, index) => (
                  <tr key={index}>
                    <td className="border px-4 py-2">{item.description}</td>
                    <td className="border px-4 py-2">{item.quantity}</td>
                    <td className="border px-4 py-2">{item.unit_price}</td>
                    <td className="border px-4 py-2">{item.total}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          ) : (
            <p className="mt-4">No items found in receipt.</p>
          )}
        </div>
      </div>
    </div>
  );
};
