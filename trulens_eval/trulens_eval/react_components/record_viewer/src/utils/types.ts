export interface AppJSONRaw {
  app_id: string;
  feedback_definitions: [];
  feedback_mode: string;
  root_class: {
    name: string;
    module: {
      package_name: string;
      module_name: string;
    };
    bases: null;
  };
  app: unknown;
}

export interface PerfJSONRaw {
  start_time: string;
  end_time: string;
}

export interface ModuleJSONRaw {
  package_name: string;
  module_name: string;
}

export interface StackJSONRaw {
  path: string; /* {
    path: ({ item_or_attribute: string } | { index: number })[];
  };*/
  method: {
    name: string;
    obj: {
      id: number;
      cls: {
        name: string;
        module: ModuleJSONRaw;
        bases: null;
      };
    };
  };
}

export interface CallJSONRaw {
  stack: StackJSONRaw[];
  args:
    | {
        _self: {
          _streaming: boolean;
        };

        kwargs: Record<string, unknown>;
      }
    | {
        str_or_query_bundle:
          | string
          | {
              custom_embedding_strs: null;
              dataclass_json_config: null;
              embedding: number[];
              embedding_strs: string[];
              query_str: string;
            };
      }
    | {
        prompt: {
          metadata: Record<string, unknown>;
          original_template:
            | string
            | {
                __tru_property_error: {
                  cls: {
                    name: string;
                    bases: null;
                    module: ModuleJSONRaw;
                  };
                  id: number;
                  init_bindings: {
                    args: string[];
                    kwargs: Record<string, string>;
                  };
                };
              };
          output_parser: null;
          partial_dict: Record<string, string>;
          prompt_kwargs: Record<string, string>;
          prompt_type: string;
          stop_token: null;
        };
        prompt_args: Record<string, string>;
      };

  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  rets: any;
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  error: any;
  perf: PerfJSONRaw;
  pid: number;
  tid: number;
}

export interface RecordJSONRaw {
  record_id: string;
  app_id: string;
  cost: {
    n_requests: number;
    n_successful_requests: number;
    n_classes: number;
    n_tokens: number;
    n_prompt_tokens: number;
    n_completion_tokens: number;
    cost: number;
  };
  perf: PerfJSONRaw;
  ts: string;
  tags: string;
  main_input: string;
  main_output: string;
  main_error: string;
  calls: CallJSONRaw[];
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  [others: string]: any;
}

export interface DataRaw {
  app_json: AppJSONRaw;
  record_json: RecordJSONRaw;
}

export interface StackTreeNode {
  children: StackTreeNode[];
  name: string;
  path?: string;
  methodName?: string;
  startTime?: Date;
  endTime?: Date;
  raw?: CallJSONRaw;
  id?: number;
  parentNodes: StackTreeNode[];
}
