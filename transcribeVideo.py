from youtube_transcript_api import YouTubeTranscriptApi

video = 'EH51mAFpcEQ'
srt = YouTubeTranscriptApi.get_transcript(video)
file = open(f'transcripts/{video}.txt', 'w')
for i in srt:
    if i['text'].endswith(']') and i['text'].startswith('['): continue
    file.write(f'{i["text"]}\n')
file.close()