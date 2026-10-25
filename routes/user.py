from fastapi import APIRouter, Request, UploadFile, File
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
import os
from moviepy.editor import VideoFileClip
import assemblyai as aai
from transformers import BartTokenizer, BartForConditionalGeneration
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
chunks = []
index = None

aai.settings.api_key = os.getenv("ASSEMBLY_API")
diarized_text = ""

router = APIRouter()
templates = Jinja2Templates(directory="templates")

def build_history(chat_history, max_turns=5):
    """
    Convert conversation history into a text format.

    Only the most recent `max_turns` conversations
    are included to prevent the prompt from becoming
    too large.
    """

    recent_history = chat_history[-max_turns:]

    history = ""

    for chat in recent_history:
        history += f"User: {chat['user']}\n"
        history += f"Assistant: {chat['assistant']}\n"

    return history

def create_chunks(text, max_words=300, overlap_turns=2):

    turns = [
        turn.strip()
        for turn in text.split("\n")
        if turn.strip()
    ]

    chunks = []
    current_chunk = []
    current_words = 0

    for i, turn in enumerate(turns, start=1):

        print("Turn:", i, turn)

        words = len(turn.split())

        if (
            current_chunk
            and current_words + words > max_words
        ):
            chunks.append("\n".join(current_chunk))

            # Keep last few turns for overlap
            current_chunk = current_chunk[-overlap_turns:]

            current_words = sum(
                len(x.split())
                for x in current_chunk
            )

        current_chunk.append(turn)
        current_words += words

    if current_chunk:
        chunks.append("\n".join(current_chunk))

    return chunks

def rewrite_query(question, chat_history):

    history = build_history(
        chat_history,
        max_turns=5
    )

    # If there is no previous conversation,
    # there is nothing to rewrite.
    if not history.strip():
        return question

    prompt = f"""
Rewrite the user's current question as a standalone question.

Use the conversation history to resolve references such as:
he, she, it, they, this, that, previous question, etc.

Do not answer the question.

Return ONLY the rewritten question.

Conversation history:
{history}

Current question:
{question}

Rewritten question:
"""

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True,
        max_length=512
    )

    output = model.generate(
        **inputs,
        max_length=100
    )

    rewritten_question = tokenizer.decode(
        output[0],
        skip_special_tokens=True
    )

    return rewritten_question.strip()

@router.get("/", response_class=HTMLResponse)
def read_root(request: Request):
    return  templates.TemplateResponse(
    request=request,
    name="landing.html"
)


@router.get("/landing", response_class=HTMLResponse)
async def lanpage(request: Request):
    return templates.TemplateResponse(
    request=request,
    name="home.html"
)


model_path = "model_folder"

model = BartForConditionalGeneration.from_pretrained(model_path)
tokenizer = BartTokenizer.from_pretrained(model_path)


@router.post("/upload", response_class=HTMLResponse)
def upload(request: Request, response_class=HTMLResponse, video: UploadFile = File(...)):
    try:
        video_path = f'uploads/{video.filename}'
        with open(video_path, 'wb') as s:
            s.write(video.file.read())
        video_clip = VideoFileClip(video_path)
        audio_clip = video_clip.audio

        audio_filename = f"uploads/{video.filename.split('.')[0]}_audio.wav"
        audio_clip.write_audiofile(audio_filename, codec='pcm_s16le')

        video_clip.close()
        audio_clip.close()
        print("Audio Extraction Successful")

        config = aai.TranscriptionConfig(speaker_labels=True)
        transcriber = aai.Transcriber()
        transcript = transcriber.transcribe(audio_filename, config=config)

        # all_speaker_utterances = []
        # 1. Split transcript
        global chunks, index, diarized_text

        diarized_text = ""

        for utterance in transcript.utterances:
            diarized_text += (
                f"Speaker {utterance.speaker}: "
                f"{utterance.text}\n"
            )

        chunks = create_chunks(diarized_text, 150, 1)

        # 2. Create embeddings
        embeddings = embedding_model.encode(
            chunks,
            normalize_embeddings=True
        )

        # 3. Convert embeddings to float32
        embeddings = np.array(
            embeddings
        ).astype("float32")

        # 4. Create FAISS index
        index = faiss.IndexFlatIP(
            embeddings.shape[1]
        )

        # 5. Add embeddings to FAISS
        index.add(embeddings)

        print(f"Created {len(chunks)} chunks")
        print(f"FAISS index contains {index.ntotal} vectors")

        return templates.TemplateResponse("interact.html", {"request": request,"diarized_text":diarized_text})

    except Exception as se:
        print(se)


# @router.post("/interact", response_class=JSONResponse)
# async def interact_model(user_input: dict):
#     question = user_input.get("user_input", '')
#     try:
#         # encoded_input = tokenizer(f"<s> {question} </s> {diarized_text} </s>", return_tensors="pt")
#         output = model.generate(**encoded_input, max_length=300)
#         decoded_output = tokenizer.decode(output[0], skip_special_tokens=True)
#         print(decoded_output)
#         return JSONResponse({"answer": decoded_output})

#     except Exception as se:
#         print(se)
#         return JSONResponse(
#         {"error": str(se)},
#         status_code=500
#     )
@router.post(
    "/interact",
    response_class=JSONResponse
)
async def interact_model(
    user_input: dict
):

    global chunks
    global index
    global chat_history

    question = user_input.get(
        "user_input",
        ""
    ).strip()

    # --------------------------------------------------------
    # Validate question
    # --------------------------------------------------------

    if not question:

        return JSONResponse(
            {
                "error": "Question cannot be empty"
            },
            status_code=400
        )

    # --------------------------------------------------------
    # Validate FAISS index
    # --------------------------------------------------------

    if index is None or not chunks:

        return JSONResponse(
            {
                "error": "Please upload a video first."
            },
            status_code=400
        )

    try:

        # ====================================================
        # STEP 1: BUILD CONVERSATION HISTORY
        # ====================================================

        history = build_history(
            chat_history,
            max_turns=5
        )

        print("\n==============================")
        print("Conversation History:")
        print("==============================")
        print(history)

        # ====================================================
        # STEP 2: REWRITE FOLLOW-UP QUESTION
        # ====================================================

        rewritten_question = rewrite_query(
            question,
            chat_history
        )

        print("\nOriginal Question:")
        print(question)

        print("\nRewritten Question:")
        print(rewritten_question)

        # ====================================================
        # STEP 3: CREATE QUESTION EMBEDDING
        # ====================================================

        question_embedding = embedding_model.encode(
            [rewritten_question],
            normalize_embeddings=True
        )

        # IMPORTANT:
        # Use question_embedding here.
        # Your previous code incorrectly used query_embedding.

        query_embedding = np.array(
            question_embedding
        ).astype("float32")

        # ====================================================
        # STEP 4: SEARCH FAISS
        # ====================================================

        k = min(
            2,
            len(chunks)
        )

        distances, indices = index.search(
            query_embedding,
            k
        )

        print("\n==============================")
        print("FAISS Results")
        print("==============================")

        for score, idx in zip(
            distances[0],
            indices[0]
        ):

            print(
                "\nScore:",
                score
            )

            print(
                "Chunk index:",
                idx
            )

            print(
                "Chunk:"
            )

            print(
                chunks[idx]
            )

        # ====================================================
        # STEP 5: RETRIEVE RELEVANT CHUNKS
        # ====================================================

        retrieved_chunks = [
            chunks[i]
            for i in indices[0]
            if i >= 0
        ]

        # ====================================================
        # STEP 6: CREATE CONTEXT
        # ====================================================

        context = "\n\n".join(
            retrieved_chunks
        )

        print("\n==============================")
        print("Retrieved Context:")
        print("==============================")

        print(context)

        # ====================================================
        # STEP 7: BUILD BART INPUT
        # ====================================================

        bart_input = f"""
        Conversation history:
        {history}

        Relevant transcript:
        {context}

        Current question:
        {question}

        Answer the current question using the relevant transcript.
        If the answer is not present in the transcript, say that the information is not available in the transcript.
        """

        # ====================================================
        # STEP 8: TOKENIZE
        # ====================================================

        encoded_input = tokenizer(
            bart_input,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )

        # ====================================================
        # STEP 9: GENERATE ANSWER
        # ====================================================

        output = model.generate(
            **encoded_input,
            max_length=300
        )

        # ====================================================
        # STEP 10: DECODE
        # ====================================================

        decoded_output = tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

        # ====================================================
        # STEP 11: SAVE CONVERSATION MEMORY
        # ====================================================

        chat_history.append(
            {
                "user": question,
                "assistant": decoded_output
            }
        )

        print("\n==============================")
        print("Answer:")
        print("==============================")

        print(decoded_output)

        # ====================================================
        # STEP 12: RETURN RESPONSE
        # ====================================================

        return JSONResponse(
            {
                "answer": decoded_output,
                "rewritten_question": rewritten_question,
                "retrieved_chunks": retrieved_chunks
            }
        )

    except Exception as se:

        print(
            "Interaction Error:",
            se
        )

        return JSONResponse(
            {
                "error": str(se)
            },
            status_code=500
        )