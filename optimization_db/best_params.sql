WITH Study AS (
    SELECT 
        trial_values.*
    FROM 
        trial_values 
    INNER JOIN 
        trials ON trial_values.trial_id = trials.trial_id 
    WHERE 
        trials.study_id = (SELECT MAX(study_id) FROM studies)
)
SELECT 
    Study.trial_id, 
    param_name, 
    param_value, 
    distribution_json, 
    value
FROM 
    Study 
INNER JOIN 
    trial_params ON Study.trial_id = trial_params.trial_id
WHERE 
    value IN (
        SELECT value 
        FROM Study 
        ORDER BY value ASC 
        LIMIT 3
    );
