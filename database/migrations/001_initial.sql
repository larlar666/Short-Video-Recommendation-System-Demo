-- Private application schema: do not add it to Supabase's exposed API schemas.
CREATE SCHEMA IF NOT EXISTS recommender;
REVOKE ALL ON SCHEMA recommender FROM PUBLIC;

CREATE TABLE recommender.videos (
    id text PRIMARY KEY,
    "desc" text NOT NULL DEFAULT '',
    challenges jsonb NOT NULL DEFAULT '[]',
    duration double precision NOT NULL DEFAULT 0 CHECK (duration >= 0),
    digg_count bigint NOT NULL DEFAULT 0 CHECK (digg_count >= 0),
    comment_count bigint NOT NULL DEFAULT 0 CHECK (comment_count >= 0),
    play_count bigint NOT NULL DEFAULT 0 CHECK (play_count >= 0),
    share_count bigint NOT NULL DEFAULT 0 CHECK (share_count >= 0),
    collect_count bigint NOT NULL DEFAULT 0 CHECK (collect_count >= 0),
    music_id text,
    user_id text, -- original creator ID, not a simulated app user
    poi_name text,
    city text,
    poi_category text,
    preference_tags jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(preference_tags) = 'array'),
    fields jsonb NOT NULL DEFAULT '[]' CHECK (jsonb_typeof(fields) = 'array')
);
CREATE INDEX videos_fields_idx ON recommender.videos USING gin (fields);
CREATE INDEX videos_play_count_idx ON recommender.videos (play_count DESC);

CREATE TABLE recommender.users (
    user_id bigint PRIMARY KEY,
    primary_field text NOT NULL,
    gamma double precision NOT NULL CHECK (gamma >= 0),
    interest_fields jsonb NOT NULL CHECK (jsonb_typeof(interest_fields) = 'array'),
    interest_tags jsonb NOT NULL CHECK (jsonb_typeof(interest_tags) = 'array'),
    total_watched integer NOT NULL CHECK (total_watched >= 0),
    total_liked integer NOT NULL CHECK (total_liked >= 0)
);
CREATE INDEX users_primary_field_idx ON recommender.users (primary_field);

CREATE TABLE recommender.video_popularity (
    id text PRIMARY KEY REFERENCES recommender.videos(id),
    "H_behavior" double precision NOT NULL,
    "Q_valid" boolean NOT NULL,
    "Q" double precision,
    "R" double precision NOT NULL,
    "H0" double precision NOT NULL,
    "H0_norm" double precision NOT NULL CHECK ("H0_norm" BETWEEN 0 AND 1)
);

CREATE TABLE recommender.watch_events (
    round integer NOT NULL CHECK (round > 0),
    user_id bigint NOT NULL REFERENCES recommender.users(user_id),
    video_id text NOT NULL REFERENCES recommender.videos(id),
    event_type text NOT NULL CHECK (event_type IN ('watch', 'like')),
    PRIMARY KEY (user_id, round, video_id, event_type)
);
CREATE INDEX watch_events_video_round_idx ON recommender.watch_events (video_id, round)
    WHERE event_type = 'watch';

-- Each prediction run has variable horizons; retain its named curve values as JSON.
CREATE TABLE recommender.video_predictions (
    video_id text PRIMARY KEY REFERENCES recommender.videos(id),
    metrics jsonb NOT NULL CHECK (jsonb_typeof(metrics) = 'object')
);
CREATE TABLE recommender.video_clusters (
    video_id text PRIMARY KEY REFERENCES recommender.videos(id),
    cluster_id integer NOT NULL CHECK (cluster_id >= 0)
);
CREATE INDEX video_clusters_cluster_idx ON recommender.video_clusters (cluster_id);
CREATE TABLE recommender.user_clusters (
    user_id bigint PRIMARY KEY REFERENCES recommender.users(user_id),
    cluster_id integer NOT NULL CHECK (cluster_id >= 0),
    primary_field text NOT NULL
);
CREATE INDEX user_clusters_cluster_idx ON recommender.user_clusters (cluster_id);

CREATE TABLE recommender.import_runs (
    dataset text NOT NULL,
    sha256 text NOT NULL,
    row_count bigint NOT NULL CHECK (row_count >= 0),
    imported_at timestamptz NOT NULL DEFAULT now(),
    PRIMARY KEY (dataset, sha256)
);

-- Browser clients have no policies or grants. Only the Python server connects.
ALTER TABLE recommender.videos ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.users ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.video_popularity ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.watch_events ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.video_predictions ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.video_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.user_clusters ENABLE ROW LEVEL SECURITY;
ALTER TABLE recommender.import_runs ENABLE ROW LEVEL SECURITY;
REVOKE ALL ON ALL TABLES IN SCHEMA recommender FROM PUBLIC;
