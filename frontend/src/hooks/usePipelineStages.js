import { useEffect, useState } from "react";
import { fetchPipelineStages } from "../api/client.js";

export function usePipelineStages() {
  const [stages, setStages] = useState([]);

  useEffect(() => {
    let cancelled = false;

    fetchPipelineStages()
      .then((data) => {
        if (!cancelled) setStages(data.stages || []);
      })
      .catch(() => {
        if (!cancelled) setStages([]);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return stages;
}
