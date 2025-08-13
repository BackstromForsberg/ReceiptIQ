# Receipt IQ

Receipt IQ is a proof of concept project using

- Strands / OCR
- Next JS with Typescript
- Python / Flask
- Docker

The purpose of the project is to ingest a pdf of any receipt and to return JSON to the front end. Optionally, using agentic AI, you can translate the receipt contents into a foreign language and/or get live foreign currency conversion.

## Required Dependencies

To Run:

- Docker

To Develop:

- Docker
- Python / pip3
- Node / npm

## Installation

Dockerfiles take care of installation if running as proof of concept

## Usage

Run Project w/ Docker:

- Have Docker Running
- Run `docker compose up` from root.
- Flask API will be served on port 5001
- UI will be served on port 3000

## Contributing

Pull requests are welcome for members of the appropriate development team. For any changes, please open an issue first to discuss what you would like to change.

## License

[MIT](https://choosealicense.com/licenses/mit/)
