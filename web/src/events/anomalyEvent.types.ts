/**
 * Generated from contracts/events/anomaly_event.schema.json via json-schema-to-typescript.
 * Run `npm run gen:events` to regenerate after the schema changes. Do not hand-edit.
 */

/**
 * Published by the AI Engine to the Web Platform. Carries no model identity (spec §3.3).
 */
export interface SentinelAIAnomalyEvent {
  schema_version: 1;
  event_id: string;
  camera_id: string;
  occurred_at: number;
  reason:
    | 'new_salient_track'
    | 'scene_change'
    | 'dwell_exceeded'
    | 'speed_anomaly'
    | 'track_count_spike'
    | 'periodic_summary'
    | 'user_requested';
  threat_score: number;
  severity: 'info' | 'low' | 'medium' | 'high' | 'critical';
  description: string;
  suggested_action: string;
  labels: string[];
  track_ids: number[];
  keyframe_uri?: string | null;
  clip_uri?: string | null;
  description_unavailable: boolean;
  metadata: {
    [k: string]: string;
  };
}
