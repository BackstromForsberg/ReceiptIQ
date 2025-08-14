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
  isLoading: boolean;
};

export const ReceiptOutput = ({ ocrOutput, isLoading }: Props) => {
  if (isLoading) {
    return (
      <div className="flex items-center justify-center w-200">
        <div className="h-6 w-6 animate-spin rounded-full border-2 border-slate-500 border-t-slate-200"></div>
      </div>
    );
  }

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
      {/* Header */}
      <div className="mb-4">
        <h2 className="text-xl font-semibold text-slate-100 tracking-tight">
          {parsed_data["Company Name"]
            ? `${parsed_data["Company Name"]} — `
            : ""}
          Receipt OCR Output
        </h2>
        <p className="mt-1 text-sm text-slate-400">Date: {parsed_data.Date}</p>
      </div>

      {/* Stat sections */}
      <div className="grid gap-4 md:grid-cols-2">
        {/* High Level Financials */}
        <section className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4">
          <h3 className="text-sm font-medium text-slate-300">
            High Level Financials
          </h3>
          <dl className="mt-3 divide-y divide-slate-700/60">
            <div className="flex items-center justify-between py-2">
              <dt className="text-slate-400">Subtotal</dt>
              <dd className="font-mono tabular-nums text-slate-100">
                {parsed_data.Subtotal}
              </dd>
            </div>
            <div className="flex items-center justify-between py-2">
              <dt className="text-slate-400">Tax</dt>
              <dd className="font-mono tabular-nums text-slate-100">
                {parsed_data["Sales Tax"]}
              </dd>
            </div>
            <div className="flex items-center justify-between py-2">
              <dt className="text-slate-400">Total</dt>
              <dd className="font-semibold font-mono tabular-nums text-slate-100">
                {parsed_data.Total}
              </dd>
            </div>
          </dl>
        </section>

        {/* Scan Results */}
        <section className="rounded-xl border border-slate-700/60 bg-slate-900/40 p-4">
          <h3 className="text-sm font-medium text-slate-300">Scan Results</h3>
          <div className="mt-3 space-y-2">
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Confidence</span>
              <span
                className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium
                         bg-emerald-500/15 text-emerald-300 ring-1 ring-emerald-500/30"
              >
                {confidence}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Extraction Method</span>
              <span
                className="inline-flex items-center rounded-full px-2.5 py-0.5 text-xs font-medium
                         bg-slate-500/15 text-slate-200 ring-1 ring-slate-400/30"
              >
                {extraction_method}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-slate-400">Processing Time</span>
              <span className="font-mono tabular-nums text-slate-100">
                {processing_time.toFixed(5)} seconds
              </span>
            </div>
          </div>
        </section>
      </div>

      <div>
        {parsed_data.Items && parsed_data.Items.length > 0 ? (
          <table className="mt-4 w-full table-auto border-collapse text-slate-200">
            <thead className="bg-slate-900/80">
              <tr>
                <th className="px-4 py-2 text-left font-semibold">Category</th>
                <th className="px-4 py-2 text-left font-semibold">
                  Description
                </th>
                <th className="px-4 py-2 text-right font-semibold">Quantity</th>
                <th className="px-4 py-2 text-right font-semibold">
                  Unit Price
                </th>
                <th className="px-4 py-2 text-right font-semibold">Total</th>
              </tr>
            </thead>
            <tbody>
              {parsed_data.Items.map((item, index) => (
                <tr
                  key={index}
                  className={`${
                    index % 2 === 0 ? "bg-slate-700/50" : "bg-slate-800/70"
                  } hover:bg-slate-600/60 transition-colors`}
                >
                  <td className="px-4 py-2">{item.category}</td>
                  <td className="px-4 py-2">{item.description}</td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {item.quantity}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {item.unit_price}
                  </td>
                  <td className="px-4 py-2 text-right tabular-nums">
                    {item.total}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="mt-4">No items found in receipt.</p>
        )}
      </div>
    </div>
  );
};
