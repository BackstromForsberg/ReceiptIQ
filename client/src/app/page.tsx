import Image from "next/image";
import { GridLayout } from "./components/GridLayout";
import { getHello } from "./actions";

export default async function Home() {
  const message = await getHello();

  return (
    <div className="font-sans flex flex-col items-center justify-items-center min-h-screen p-20 gap-48 px-20 mt-40">
      <main className="flex flex-col gap-[12px] row-start-2 items-center sm:items-start">
        <GridLayout />
        <p>Connected to Flask API: {message}</p>
      </main>
      <footer className="row-start-3 flex flex-wrap items-center gap-20 justify-center">
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
            width={36}
            height={36}
          />
          GitHub
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://ollama.com/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/ollama-white.png"
            alt="File icon"
            width={36}
            height={36}
          />
          Powered by Ollama LLM
        </a>
        <a
          className="flex items-center gap-2 hover:underline hover:underline-offset-4"
          href="https://github.com/goagiq/VisionOCR/"
          target="_blank"
          rel="noopener noreferrer"
        >
          <Image
            aria-hidden
            src="/file.svg"
            alt="File icon"
            width={36}
            height={36}
          />
          Powered by Larry (Vision OCR)
        </a>
      </footer>
    </div>
  );
}
