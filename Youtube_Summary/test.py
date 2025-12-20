from youtube_transcript_api import YouTubeTranscriptApi

video_id = "O2gerCxEXvc"  # your sample video

transcript = YouTubeTranscriptApi.get_transcript(video_id, languages=["en"])
print(len(transcript), "chunks")
print(transcript[0])