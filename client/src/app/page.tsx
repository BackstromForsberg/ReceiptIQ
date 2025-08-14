import Image from "next/image";
import { GridLayout } from "./components/GridLayout";
import { getHello } from "./actions";

export default async function Home() {
  const message = await getHello();

  return (
    <div className="font-sans grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen pb-20 gap-16 sm:p-20">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <GridLayout />
        <p>Connected to Flask API: {message}</p>
      </main>
      <footer className="row-start-3 flex gap-[24px] flex-wrap items-center justify-center">
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://github.com/BackstromForsberg/ReceiptIQ"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={16}
            height={16}
          />
          Powered by Larry & Ollama
        </a>
      </footer>
    </div>
  );
}
