CREATE OR REPLACE FUNCTION public.notify_new_transcript()
RETURNS trigger AS $$
BEGIN
  PERFORM pg_notify('meeting_transcript_insert', NEW.id::text);
  RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DROP TRIGGER IF EXISTS trg_notify_new_transcript
ON public."myApp_meetingtranscript";

CREATE TRIGGER trg_notify_new_transcript
AFTER INSERT ON public."myApp_meetingtranscript"
FOR EACH ROW
EXECUTE FUNCTION public.notify_new_transcript();
