import Image from "next/image";

export default function Home() {
  return (
    <div className="font-sans grid grid-rows-[20px_1fr_20px] items-center justify-items-center min-h-screen p-8 pb-20 gap-16 sm:p-20">
      <main className="flex flex-col gap-[32px] row-start-2 items-center sm:items-start">
        <h1 className="text-4xl font-bold text-center sm:text-left">
          Welcome to Next.js with Tailwind CSS!
        </h1>
        <p className="text-lg text-gray-700 text-center sm:text-left">
          This is a simple starter template for building applications with
          Next.js and Tailwind CSS. Explore the links below to learn more.
        </p>
        <div className="flex flex-col gap-4 sm:flex-row">
          <a
            className="inline-block px-6 py-3 bg-blue-600 text-white rounded hover:bg-blue-700 transition"
            href="https://nextjs.org/docs"
            target="_blank"
            rel="noopener noreferrer"
          >
            Documentation
          </a>
          <a
            className="inline-block px-6 py-3 bg-green-600 text-white rounded hover:bg-green-700 transition"
            href="https://tailwindcss.com/docs"
            target="_blank"
            rel="noopener noreferrer"
          >
            Tailwind CSS Docs
          </a>
        </div>
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
          GitHub Repository Link
        </a>
      </footer>
    </div>
  );
}
