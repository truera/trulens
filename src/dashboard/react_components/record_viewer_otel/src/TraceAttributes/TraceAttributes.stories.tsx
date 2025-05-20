import type { Meta, StoryObj } from '@storybook/react';
import { TraceAttributes } from './TraceAttributes';
import { SpanAttributes, SpanType } from '../constants/span';

const meta = {
  title: 'Components/TraceAttributes',
  component: TraceAttributes,
  parameters: {
    layout: 'centered',
  },
} satisfies Meta<typeof TraceAttributes>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    attributes: {},
  },
};

export const GenerationAttributes: Story = {
  args: {
    attributes: {
      [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION,
      [SpanAttributes.RECORD_ID]: 'record-789012',
      [SpanAttributes.GENERATION_MODEL_NAME]: 'gpt-4',
      [SpanAttributes.GENERATION_MODEL_TYPE]: 'chat',
      [SpanAttributes.GENERATION_INPUT_TOKEN_COUNT]: 150,
      [SpanAttributes.GENERATION_OUTPUT_TOKEN_COUNT]: 320,
      [SpanAttributes.GENERATION_TEMPERATURE]: 0.7,
      [SpanAttributes.GENERATION_INPUT_MESSAGES]: JSON.stringify([
        { role: 'system', content: 'You are a helpful assistant.' },
        { role: 'user', content: 'How do I build a React component?' },
      ]),
      [SpanAttributes.GENERATION_OUTPUT_MESSAGES]: JSON.stringify([
        {
          role: 'assistant',
          content: 'To build a React component, first create a new file with a .jsx or .tsx extension...',
        },
      ]),
    },
  },
};

export const RetrievalAttributes: Story = {
  args: {
    attributes: {
      [SpanAttributes.SPAN_TYPE]: SpanType.RETRIEVAL,
      [SpanAttributes.RECORD_ID]: 'record-345678',
      [SpanAttributes.RETRIEVAL_QUERY_TEXT]: 'How to implement RAG in an application?',
      [SpanAttributes.RETRIEVAL_DISTANCE_TYPE]: 'cosine',
      [SpanAttributes.RETRIEVAL_NUM_CONTEXTS]: 3,
      [SpanAttributes.RETRIEVAL_RETRIEVED_CONTEXTS]: JSON.stringify([
        'Retrieval-Augmented Generation (RAG) is a technique that...',
        'To implement RAG, you need to first create an embedding of your documents...',
        'The key components of a RAG system include a vector database...',
      ]),
      [SpanAttributes.RETRIEVAL_RETRIEVED_SCORES]: JSON.stringify([0.92, 0.85, 0.78]),
    },
  },
};

export const CostAttributes: Story = {
  args: {
    attributes: {
      [SpanAttributes.SPAN_TYPE]: SpanType.GENERATION,
      [SpanAttributes.RECORD_ID]: 'record-901234',
      [SpanAttributes.COST_COST]: 0.0567,
      [SpanAttributes.COST_COST_CURRENCY]: 'USD',
      [SpanAttributes.COST_MODEL]: 'gpt-4-turbo',
      [SpanAttributes.COST_NUM_TOKENS]: 950,
      [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 250,
      [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 700,
    },
  },
};

export const EvalAttributes: Story = {
  args: {
    attributes: {
      [SpanAttributes.SPAN_TYPE]: SpanType.EVAL,
      [SpanAttributes.RECORD_ID]: 'record-567890',
      [SpanAttributes.EVAL_FEEDBACK_NAME]: 'relevance',
      [SpanAttributes.EVAL_CRITERIA]: 'Evaluate if the response is relevant to the query',
      [SpanAttributes.EVAL_SCORE]: 0.85,
      [SpanAttributes.EVAL_EXPLANATION]:
        'The response addresses the core question but misses some context about implementation details.',
    },
  },
};

export const ComplexAttributes: Story = {
  args: {
    attributes: {
      [SpanAttributes.SPAN_TYPE]: SpanType.RECORD_ROOT,
      [SpanAttributes.APP_NAME]: 'AI Customer Service',
      [SpanAttributes.APP_VERSION]: '2.1.3',
      [SpanAttributes.RECORD_ID]: 'record-123890',
      [SpanAttributes.DOMAIN]: 'customer-support',
      [SpanAttributes.RECORD_ROOT_INPUT]: 'How do I reset my password?',
      [SpanAttributes.RECORD_ROOT_OUTPUT]:
        'To reset your password, please go to the login page and click on "Forgot Password"...',
      [SpanAttributes.COST_COST]: 0.0321,
      [SpanAttributes.COST_COST_CURRENCY]: 'USD',
      [SpanAttributes.COST_NUM_TOKENS]: 425,
      [SpanAttributes.COST_NUM_PROMPT_TOKENS]: 125,
      [SpanAttributes.COST_NUM_COMPLETION_TOKENS]: 300,
    },
  },
};
