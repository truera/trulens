import type { Meta, StoryObj } from '@storybook/react';
import { TraceContent } from './TraceContent';

const meta = {
  title: 'Components/TraceContent',
  component: TraceContent,
  parameters: {
    layout: 'centered',
  },
} satisfies Meta<typeof TraceContent>;

export default meta;
type Story = StoryObj<typeof meta>;

export const Empty: Story = {
  args: {
    rawValue: null,
  },
};

export const EmptyString: Story = {
  args: {
    rawValue: '',
  },
};

export const SimpleString: Story = {
  args: {
    rawValue: 'This is a simple string value',
  },
};

export const LongString: Story = {
  args: {
    rawValue: 'This is a very long string that contains a lot of text and should be displayed properly. '.repeat(10),
  },
};

export const JsonString: Story = {
  args: {
    rawValue: JSON.stringify({
      name: 'Test User',
      role: 'admin',
      permissions: ['read', 'write', 'execute'],
      metadata: {
        lastLogin: '2023-07-15T12:30:45Z',
        createdAt: '2022-01-01T00:00:00Z',
      },
    }),
  },
};

export const ArrayOfStrings: Story = {
  args: {
    rawValue: ['Short string', 'Another short string', 'Third string'],
  },
};

export const ArrayOfLongStrings: Story = {
  args: {
    rawValue: [
      'This is a very long string that exceeds the maximum length and should trigger the JSON viewer. '.repeat(5),
      'Another long string that also exceeds the maximum length limit set in the component. '.repeat(5),
      'A third long string just to make sure we have multiple items in our test array. '.repeat(5),
    ],
  },
};

export const SimpleObject: Story = {
  args: {
    rawValue: {
      id: 12345,
      name: 'Test Object',
      active: true,
      tags: ['important', 'test'],
    },
  },
};

export const NestedObject: Story = {
  args: {
    rawValue: {
      user: {
        id: 'user-123',
        name: 'Test User',
        email: 'test@example.com',
      },
      request: {
        type: 'query',
        content: 'How do I reset my password?',
        timestamp: '2023-07-15T12:30:45Z',
      },
      response: {
        status: 'success',
        data: {
          instructions: 'To reset your password, please go to the login page and click on "Forgot Password"...',
          links: ['https://example.com/reset', 'https://example.com/help'],
        },
      },
    },
  },
};

export const NumberValue: Story = {
  args: {
    rawValue: 42,
  },
};

export const BooleanValue: Story = {
  args: {
    rawValue: true,
  },
};
