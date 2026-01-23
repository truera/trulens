import type { Meta, StoryObj } from '@storybook/react';
import JSONViewer from './JSONViewer';

type Story = StoryObj<typeof JSONViewer>;

const meta: Meta<typeof JSONViewer> = {
  title: 'Components/JSONViewer',
  component: JSONViewer,
  args: {
    src: { sample: 'data' },
  },
};

export default meta;

export const Default: Story = {};

export const WithNestedObject: Story = {
  args: {
    src: {
      string: 'value',
      number: 42,
      boolean: true,
      null: null,
      array: [1, 2, 3],
      object: {
        nested: 'property',
        anotherNested: {
          deeplyNested: true,
        },
      },
    },
  },
};

export const WithArray: Story = {
  args: {
    src: [
      { id: 1, name: 'Item 1' },
      { id: 2, name: 'Item 2' },
      { id: 3, name: 'Item 3' },
    ],
  },
};

export const WithLongString: Story = {
  args: {
    src: {
      description:
        'This is a very long string that will be collapsed after 140 characters according to the component settings. It demonstrates how the JSONViewer handles lengthy text content within the JSON structure. The component should automatically truncate this text when displayed initially.',
      shortValue: 'This is a shorter value',
    },
  },
};

export const Empty: Story = {
  args: {
    src: {},
  },
};
